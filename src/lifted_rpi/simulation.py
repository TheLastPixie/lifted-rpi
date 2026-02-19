"""
Simulation utilities for validation and data generation.

``eps_MRPI``
    Outer epsilon-approximation of the minimal RPI set using Rakovic
    et al. (2005, Algorithm 1).  Requires pytope with pycddlib<3.

``calculate_realistic_disturbance``
    Velocity-dependent drag + input inefficiency + Gaussian noise.
    Returns a 4-vector disturbance injected into acceleration rows.

``generate_trajectory``
    Reference trajectory generators (linear, circular, figure-8,
    sinusoidal, spiral) for MPC benchmarks.

``simulate_trajectory_with_realistic_drag``
    Closed-loop MPC simulation with the drag disturbance model.
    Used to generate training data for the GP learner.

"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize


# ═══════════════ ε-MRPI (Raković Algorithm 1) ═══════════════

def eps_MRPI(A, W, epsilon, s_max=20):
    """
    Outer ε-approximation of the minimal RPI set for x⁺ = Ax + w.

    Implements Algorithm 1 from Raković et al. (2005):
        F(α, s) = (1 − α)⁻¹ F_s  where  F_s = Σ_{i=0}^{s−1} Aⁱ W

    Parameters
    ----------
    A : array (n, n), strictly stable state-transition matrix
    W : pytope.Polytope, compact disturbance set containing the origin
    epsilon : float, approximation tolerance (> 0)
    s_max : int, upper bound on the number of Minkowski-sum terms

    Returns
    -------
    F_alpha_s : pytope.Polytope, the epsilon-approximation
    result : dict with 'alpha', 's', 'M', 'status', 'alpha_o_s',
             'F_s', 'eps_min'
    """
    from pytope import Polytope as plpytope

    status = -1
    m, n = A.shape
    if m != n:
        raise ValueError("A must be a square matrix")

    W.minimize_V_rep()
    F = W.A
    g = W.b
    I = g.size

    if not all(g > 0):
        raise ValueError("W does not contain the origin: g > 0 is not satisfied")

    alpha_o_s = np.full(s_max, np.nan)
    M_s_row = np.zeros((s_max, 2 * n))
    M = np.full(s_max, np.nan)
    A_pwr = np.stack([np.linalg.matrix_power(A, i) for i in range(s_max)])
    alpha_o = np.full(I, np.nan)

    s = 0
    while s < s_max - 1:
        s += 1

        for i in range(I):
            fi = F[i, :].T
            h_W_i, st = W.support(A_pwr[s].T @ fi)
            if not st.success:
                print(
                    f"Unsuccessful support evaluation h_W((A^{s})' * f_{i}): "
                    f"{st.message}"
                )
            alpha_o[i] = h_W_i / g[i]
        alpha_o_s[s] = np.max(alpha_o)
        alpha = alpha_o_s[s]

        h_W_sm1_pos_j = np.full(n, np.nan)
        h_W_sm1_neg_j = np.full(n, np.nan)
        for j in range(n):
            A_pwr_i = A_pwr[s - 1]
            h_W_sm1_pos_j[j], st_l = W.support(A_pwr_i[j])
            h_W_sm1_neg_j[j], st_r = W.support(-A_pwr_i[j])
            if not all(st.success for st in (st_l, st_r)):
                raise ValueError(
                    f"Unsuccessful support evaluation in direction of row {j} "
                    f"of A^{s - 1} (s = {s})"
                )

        M_s_row[s] = M_s_row[s - 1] + np.concatenate(
            (h_W_sm1_pos_j, h_W_sm1_neg_j)
        )
        M[s] = np.max(M_s_row[s])

        if alpha <= epsilon / (epsilon + M[s]):
            status = 0
            break

    s_final = s

    F_s = np.full(s_final + 1, plpytope(np.zeros((1, n))))
    for ss in range(1, s_final + 1):
        F_s[ss] = F_s[ss - 1] + A_pwr[ss - 1] * W
        F_s[ss].minimize_V_rep()

    F_alpha_s = F_s[s_final] * (1 / (1 - alpha))

    eps_min = M[s_final] * alpha / (1 - alpha)

    result = {
        "alpha": alpha,
        "s": s_final,
        "M": M[: s_final + 1],
        "status": status,
        "alpha_o_s": alpha_o_s[: s_final + 1],
        "F_s": F_s,
        "eps_min": eps_min,
    }
    return F_alpha_s, result


# ═══════════════ realistic disturbance model ═══════════════

def calculate_realistic_disturbance(
    state: np.ndarray,
    control_input: np.ndarray,
    beta1: float = 0.5,
    beta2: float = 0.2,
    mass: float = 1.0,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Velocity-dependent drag + input inefficiency + Gaussian noise.

    Returns a 4-vector disturbance [0, w_x, 0, w_y] injected into
    the acceleration rows.
    """
    vx = state[1]
    vy = state[3]
    v_magnitude = np.sqrt(vx**2 + vy**2) + 1e-6

    drag_x = -beta1 / mass * v_magnitude * vx
    drag_y = -beta1 / mass * v_magnitude * vy

    ineff_x = -beta2 / mass * control_input[0]
    ineff_y = -beta2 / mass * control_input[1]

    noise_x = np.random.normal(0, noise_std)
    noise_y = np.random.normal(0, noise_std)

    disturbance = np.zeros(4)
    disturbance[1] = drag_x + ineff_x + noise_x
    disturbance[3] = drag_y + ineff_y + noise_y
    return disturbance


# ═══════════════ trajectory generators ═══════════════

def generate_trajectory(
    trajectory_type: str,
    x_initial: np.ndarray,
    x_target: np.ndarray,
    n_steps: int,
    dt: float = 0.02,
) -> np.ndarray:
    """
    Generate reference trajectories of various types.

    Supported: 'linear', 'circular', 'figure8', 'sinusoidal', 'spiral'.
    Returns an (n_steps, 4) array [p_x, v_x, p_y, v_y].
    """
    t = np.linspace(0, 1, n_steps)
    reference_path = np.zeros((n_steps, 4))

    if trajectory_type == "linear":
        reference_path = np.linspace(x_initial, x_target, n_steps)

    elif trajectory_type == "circular":
        cx = (x_initial[0] + x_target[0]) / 2
        cy = (x_initial[2] + x_target[2]) / 2
        radius = np.linalg.norm(x_target - x_initial) / 2
        theta = np.linspace(0, 2 * np.pi, n_steps)
        reference_path[:, 0] = cx + radius * np.cos(theta)
        reference_path[:, 2] = cy + radius * np.sin(theta)
        reference_path[:, 1] = (
            -radius * np.sin(theta) * (2 * np.pi / (n_steps * dt))
        )
        reference_path[:, 3] = (
            radius * np.cos(theta) * (2 * np.pi / (n_steps * dt))
        )

    elif trajectory_type == "figure8":
        cx = (x_initial[0] + x_target[0]) / 2
        cy = (x_initial[2] + x_target[2]) / 2
        sx = np.abs(x_target[0] - x_initial[0]) / 2
        sy = np.abs(x_target[2] - x_initial[2]) / 2
        t_f8 = np.linspace(0, 2 * np.pi, n_steps)
        reference_path[:, 0] = cx + sx * np.sin(t_f8)
        reference_path[:, 2] = cy + sy * np.sin(t_f8) * np.cos(t_f8)
        reference_path[:, 1] = (
            sx * np.cos(t_f8) * (2 * np.pi / (n_steps * dt))
        )
        reference_path[:, 3] = (
            sy
            * (np.cos(t_f8) ** 2 - np.sin(t_f8) ** 2)
            * (2 * np.pi / (n_steps * dt))
        )

    elif trajectory_type == "sinusoidal":
        amp_y = np.abs(x_target[2] - x_initial[2]) / 2
        freq = 2 * np.pi / (x_target[0] - x_initial[0])
        reference_path[:, 0] = np.linspace(x_initial[0], x_target[0], n_steps)
        reference_path[:, 2] = x_initial[2] + amp_y * (
            1 + np.sin(freq * (reference_path[:, 0] - x_initial[0]))
        )
        dx_dt = (x_target[0] - x_initial[0]) / (n_steps * dt)
        reference_path[:, 1] = dx_dt
        reference_path[:, 3] = (
            amp_y * freq * np.cos(freq * (reference_path[:, 0] - x_initial[0])) * dx_dt
        )

    elif trajectory_type == "spiral":
        cx = x_initial[0]
        cy = x_initial[2]
        radius_inc = np.linalg.norm(x_target - x_initial) / (2 * np.pi)
        theta = np.linspace(0, 2 * np.pi, n_steps)
        radius = radius_inc * theta
        reference_path[:, 0] = cx + radius * np.cos(theta)
        reference_path[:, 2] = cy + radius * np.sin(theta)
        dtheta_dt = 2 * np.pi / (n_steps * dt)
        reference_path[:, 1] = (
            radius_inc * np.cos(theta) * dtheta_dt
            - radius * np.sin(theta) * dtheta_dt
        )
        reference_path[:, 3] = (
            radius_inc * np.sin(theta) * dtheta_dt
            + radius * np.cos(theta) * dtheta_dt
        )

    # fix velocity endpoints for linear; set them for others
    if trajectory_type == "linear":
        reference_path[:, 1] = np.linspace(x_initial[1], x_target[1], n_steps)
        reference_path[:, 3] = np.linspace(x_initial[3], x_target[3], n_steps)
    else:
        reference_path[0, 1] = x_initial[1]
        reference_path[0, 3] = x_initial[3]
        reference_path[-1, 1] = x_target[1]
        reference_path[-1, 3] = x_target[3]

    return reference_path


# ═══════════════ closed-loop simulation with MPC ═══════════════

def simulate_trajectory_with_realistic_drag(
    trajectory_type: str,
    x_initial: np.ndarray,
    x_target: np.ndarray,
    n_steps: int,
    L_k: int,
    A_p: np.ndarray,
    B_p: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    beta1: float = 0.5,
    beta2: float = 0.2,
    mass: float = 1.0,
    noise_std: float = 0.05,
    dt: float = 0.02,
    robust_V_constraints: Optional[List] = None,
    robust_U_constraints: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Simulate the system with realistic drag + MPC.

    Parameters
    ----------
    robust_V_constraints : list of (VSet_x, VSet_y) per horizon step, optional
        Per-step robust velocity constraint sets. Each entry is a tuple
        (V_x_set, V_y_set) whose .V attribute gives the allowed range.
        When provided, squared penalty is added for velocity violations.
    robust_U_constraints : list of (U_x_set, U_y_set) per horizon step, optional
        Per-step robust control constraint sets. Same structure as above.

    Returns (state_history, control_history, disturbance_history,
             reference_path, total_time, fps).
    """
    import time as _time

    reference_path = generate_trajectory(
        trajectory_type, x_initial, x_target, n_steps, dt=dt
    )

    state_history = []
    control_history = []
    disturbance_history = []
    current_state = x_initial.copy()

    def _mpc_cost(v, z_init, ref, step):
        z = np.zeros((4, L_k + 1))
        z[:, 0] = z_init.flatten()
        cost = 0.0
        n_ref = len(ref)
        for j in range(L_k):
            u = v[2 * j : 2 * (j + 1)]
            z[:, j + 1] = A_p @ z[:, j] + B_p @ u
            ref_idx = min(max(step + j, 0), n_ref - 1)
            x_ref = ref[ref_idx]

            # Tracking + input cost
            cost += (x_ref - z[:, j]).T @ Q @ (x_ref - z[:, j]) + u.T @ R @ u

            # Robust control constraint penalty
            if robust_U_constraints is not None and j < len(robust_U_constraints):
                u_cx, u_cy = robust_U_constraints[j]
                ctrl_dev_x = (max(0, u[0] - u_cx.V.max())
                              + max(0, u_cx.V.min() - u[0]))
                ctrl_dev_y = (max(0, u[1] - u_cy.V.max())
                              + max(0, u_cy.V.min() - u[1]))
                cost += ctrl_dev_x ** 2 + ctrl_dev_y ** 2

            # Robust velocity constraint penalty
            if robust_V_constraints is not None and j < len(robust_V_constraints):
                v_cx, v_cy = robust_V_constraints[j]
                vel_dev_x = (max(0, z[1, j + 1] - v_cx.V.max())
                             + max(0, v_cx.V.min() - z[1, j + 1]))
                vel_dev_y = (max(0, z[3, j + 1] - v_cy.V.max())
                             + max(0, v_cy.V.min() - z[3, j + 1]))
                cost += vel_dev_x ** 2 + vel_dev_y ** 2

        final_ref_idx = min(max(step + L_k, 0), n_ref - 1)
        cost += (
            (ref[final_ref_idx] - z[:, L_k]).T
            @ P
            @ (ref[final_ref_idx] - z[:, L_k])
        )
        return cost

    t_start = _time.time()
    for step in range(n_steps):
        z_init = current_state.flatten()
        result = minimize(
            _mpc_cost,
            x0=np.zeros(L_k * 2),
            args=(z_init, reference_path, step),
            method="SLSQP",
        )
        u_opt = result.x[:2]

        disturbance = calculate_realistic_disturbance(
            current_state, u_opt, beta1, beta2, mass, noise_std
        )
        current_state = A_p @ current_state + B_p @ u_opt + disturbance
        current_state = current_state.flatten()

        state_history.append(current_state.copy())
        control_history.append(u_opt)
        disturbance_history.append(disturbance)
    t_end = _time.time()

    state_history = np.array(state_history)
    control_history = np.array(control_history)
    disturbance_history = np.array(disturbance_history)
    total_time = t_end - t_start
    fps = n_steps / total_time

    return (
        state_history,
        control_history,
        disturbance_history,
        reference_path,
        total_time,
        fps,
    )


def plot_trajectory_with_disturbance(
    state_history: np.ndarray,
    control_history: np.ndarray,
    disturbance_history: np.ndarray,
    reference_path: np.ndarray,
    total_time: float,
    trajectory_type: str,
    params: Dict,
):
    """Six-panel matplotlib dashboard for a trajectory simulation."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(state_history[:, 0], state_history[:, 2], label="Actual", color="blue")
    ax.plot(
        reference_path[:, 0],
        reference_path[:, 2],
        label="Reference",
        color="orange",
        linestyle=":",
    )
    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    ax.grid(True)
    ax.set_title(f"{trajectory_type.capitalize()} Trajectory with Realistic Drag")
    ax.legend()

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(state_history[:, 1], label="v_x", color="orange")
    ax.plot(state_history[:, 3], label="v_y", color="green")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True)
    ax.set_title("Velocity Over Time")
    ax.legend()

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(control_history[:, 0], label="u_x", color="red")
    ax.plot(control_history[:, 1], label="u_y", color="purple")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Control Input (m/s²)")
    ax.grid(True)
    ax.set_title("Control Inputs")
    ax.legend()

    ax = fig.add_subplot(2, 3, 4)
    pe_x = reference_path[: len(state_history), 0] - state_history[:, 0]
    pe_y = reference_path[: len(state_history), 2] - state_history[:, 2]
    err = np.sqrt(pe_x**2 + pe_y**2)
    ax.plot(err, label="Position Error", color="brown")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Error Magnitude (m)")
    ax.grid(True)
    ax.set_title("Tracking Error")
    ax.legend()

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(disturbance_history[:, 1], label="w_x", color="darkblue")
    ax.plot(disturbance_history[:, 3], label="w_y", color="darkgreen")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Disturbance (m/s²)")
    ax.grid(True)
    ax.set_title("Disturbance Components")
    ax.legend()

    ax = fig.add_subplot(2, 3, 6)
    vmag = np.sqrt(state_history[:, 1] ** 2 + state_history[:, 3] ** 2)
    dmag = np.sqrt(disturbance_history[:, 1] ** 2 + disturbance_history[:, 3] ** 2)
    sc = ax.scatter(vmag, dmag, alpha=0.5, c=range(len(vmag)), cmap="viridis")
    plt.colorbar(sc, ax=ax, label="Time Step")
    ax.set_xlabel("Velocity Magnitude (m/s)")
    ax.set_ylabel("Disturbance Magnitude (m/s²)")
    ax.grid(True)
    ax.set_title("Velocity vs. Disturbance Magnitude")

    plt.tight_layout()

    ns = len(state_history)
    print(f"Trajectory Type: {trajectory_type}")
    print(f"Simulation completed in {total_time:.2f} seconds")
    print(f"Control Rate: {ns / total_time:.2f} Hz")
    print(f"Average Position Error: {np.mean(err):.4f} m")
    print(f"Max Position Error: {np.max(err):.4f} m")
    plt.show()


def test_realistic_drag(
    trajectory_types=None,
    n_steps: int = 300,
    beta1: float = 0.5,
    beta2: float = 0.2,
    mass: float = 1.0,
    noise_std: float = 0.05,
    L_k: int = 5,
    dt: float = 0.02,
):
    """Convenience entry-point: generate LQR, run sim, plot."""
    if trajectory_types is None:
        trajectory_types = ["linear", "circular", "figure8", "sinusoidal", "spiral"]
    elif isinstance(trajectory_types, str):
        trajectory_types = [trajectory_types]

    A_p = np.array(
        [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]
    )
    B_p = np.array(
        [[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]]
    )
    Q = np.diag([1000, 0.1, 1000, 0.1])
    R = np.diag([0.1, 0.1])
    P = solve_discrete_are(A_p, B_p, Q, R)

    x_initial = np.array([0, 0, 4, 0])
    x_target = np.array([10, 0, 10, 0])
    params = {"beta1": beta1, "beta2": beta2, "mass": mass, "noise_std": noise_std}

    for traj_type in trajectory_types:
        print(f"\nSimulating {traj_type} trajectory...")
        sh, ch, dh, rp, tt, fps = simulate_trajectory_with_realistic_drag(
            traj_type,
            x_initial,
            x_target,
            n_steps,
            L_k,
            A_p,
            B_p,
            Q,
            R,
            P,
            beta1,
            beta2,
            mass,
            noise_std,
            dt=dt,
        )
        plot_trajectory_with_disturbance(sh, ch, dh, rp, tt, traj_type, params)
