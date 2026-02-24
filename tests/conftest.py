"""Pytest configuration: block CuPy/JAX GPU backends during unit tests.

On machines where CUDA drivers are not fully functional (e.g., Windows
without matching CUDA toolkit), importing CuPy can cause access-violation
crashes within pytest.  We pre-empt this by installing import blockers
before any lifted_rpi imports happen.
"""
import sys
import types
import pathlib


class _GPUBlocker:
    """Meta-path finder that blocks CuPy and JAX imports."""

    _BLOCKED = frozenset(("cupy", "jax", "jaxlib"))

    def find_module(self, fullname, path=None):
        top = fullname.split(".")[0]
        if top in self._BLOCKED:
            return self
        return None

    def load_module(self, fullname):
        raise ImportError(f"{fullname} blocked for testing")


sys.meta_path.insert(0, _GPUBlocker())

# Ensure PYTHONPATH has src/ so lifted_rpi is importable
_src = str(pathlib.Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
