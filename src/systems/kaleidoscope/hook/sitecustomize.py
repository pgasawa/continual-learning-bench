"""Start-up hook for the Kaleidoscope experiment on Continual Learning Bench.

Loaded by putting this directory on PYTHONPATH; the logic is in
``clbench_ks_hook.py`` beside it. This file hides the interpreter's own
sitecustomize (Homebrew's Python ships one that rewrites ``sys.base_prefix``),
so it always runs that one first, exactly as Python would have. Then:

- ``CLBENCH_KS_HOOK`` unset, empty or ``0``: nothing more, and no module is
  imported.
- ``CLBENCH_KS_HOOK=1``: import ``clbench_ks_hook`` from this directory and
  install it. Any failure writes one line to stderr and exits with code 78.
- any other value: exit 78, so a misspelt switch cannot run without the hook.
"""

import os
import sys

_EXIT_CODE = 78
_HERE = os.path.dirname(os.path.realpath(__file__))


def _fail(reason):
    line = "clbench_ks_hook: FATAL, exiting 78: " + " ".join(str(reason).split())
    try:
        os.write(2, (line[:2000] + "\n").encode("utf-8", "replace"))
    finally:
        os._exit(_EXIT_CODE)


def _run_shadowed_sitecustomize():
    """Run the first sitecustomize on sys.path that is not a copy of this hook."""
    import importlib.machinery
    import importlib.util

    path = [
        entry
        for entry in sys.path
        if not os.path.isfile(os.path.join(entry or os.curdir, "clbench_ks_hook.py"))
    ]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _start_hook():
    try:
        _run_shadowed_sitecustomize()
        import clbench_ks_hook

        found = os.path.dirname(os.path.realpath(clbench_ks_hook.__file__))
        if found != _HERE:
            raise ImportError(f"clbench_ks_hook resolved to {found}, not {_HERE}")
    except BaseException as exc:
        _fail(f"{type(exc).__name__}: {exc}")
    clbench_ks_hook.install()


_switch = os.environ.get("CLBENCH_KS_HOOK", "")
if _switch == "1":
    _start_hook()
elif _switch in ("", "0"):
    _run_shadowed_sitecustomize()
else:
    _fail(f"CLBENCH_KS_HOOK={_switch!r}; expected unset, 0 or 1")
