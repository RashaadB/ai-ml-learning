# 02 Caching and checkpoint utilities

Overview

This document explains the cell that sets up file based caching, small helper functions for pickling, a simple timing decorator, and environment tuning for CPU threading.

Purpose

- Limit thread usage on macOS to avoid performance degradation.
- Create an `artifacts` directory for saved intermediates.
- Provide helpers to check for cached files and to save and load pickled objects.
- Provide a decorator to time long running functions.
- Expose a `SMOKE_TEST` flag for quick local testing.

Line by line explanation

1. `from pathlib import Path`
   - `Path` is a modern API for path manipulation that works across platforms.

2. `import json, time, pickle, os`
   - Common standard library modules: `json` for JSON handling, `time` for performance timing, `pickle` for Python object serialization, and `os` for environment variables and OS-level operations.

3. Environment variables to limit threads
   - `os.environ.setdefault("OMP_NUM_THREADS","4")`
   - `os.environ.setdefault("OPENBLAS_NUM_THREADS","4")`
   - `os.environ.setdefault("MKL_NUM_THREADS","4")`
   - `os.environ.setdefault("VECLIB_MAXIMUM_THREADS","4")`
   - `os.environ.setdefault("NUMEXPR_NUM_THREADS","4")`
   - These set the maximum number of threads used by low level numerical libraries. On macOS these libraries can spawn too many threads and slow down overall performance; setting these limits reduces contention.

4. `ARTIFACTS = Path("./artifacts")` and `ARTIFACTS.mkdir(exist_ok=True)`
   - Define a path to the `artifacts` directory and create it if it does not exist. This central location is used to save intermediate files like pickled vectorizers.

5. `def already(path):` and its body
   - Check if a path points to an existing file with nonzero size. This lets the Notebook skip expensive recomputation if a cached artifact is present.

6. `def save_pickle(obj, path):` and `def load_pickle(path):`
   - Wrappers around `pickle.dump` and `pickle.load` to persist Python objects to disk. Using these keeps save and load logic consistent across the notebook.

7. `def timed(msg):` and the decorator implementation
   - A decorator factory that prints a start message, measures execution time of a function, and prints a done message with elapsed time. Useful to annotate long running operations such as model training.

8. `SMOKE_TEST = True` and `print(f"SMOKE_TEST = {SMOKE_TEST}")`
   - A simple flag that toggles shorter runs and smaller sample sizes for quick checks. The print helps the user confirm the current mode.

Inputs and outputs

- Inputs: None directly. It configures environment variables and creates the `artifacts` directory.
- Outputs: Side effects include environment variable settings, folder creation, and printout showing the smoke test flag.

Notes and tips

- If you run into slow notebook responsiveness, lower thread counts further or run without heavy parallel operations.
- Use `ARTIFACTS` to store other persistent items so you can reload them in future runs.
- `SMOKE_TEST = True` is useful for demonstrations and quick iteration; set to False for full experiments.

---

End of 02 Caching and checkpoint utilities
