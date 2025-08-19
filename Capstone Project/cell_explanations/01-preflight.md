# 01 Preflight check

Overview

This document explains the first code cell in `capstone_project_one.ipynb`. That cell performs an environment preflight. It checks for required packages, warns about known compatibility issues, and can enforce stricter version requirements when needed.

Purpose

- Stop the notebook early if required packages are missing.
- Warn about common incompatible package combinations.
- Optionally enforce minimum versions when strict mode is enabled.

Line by line explanation

1. `#Preflight v2: fail on real breakage, warn otherwise `
   - A human readable comment describing the high level behavior of the cell.

2. `import sys, importlib, platform`
   - `sys` is used for Python runtime information. `importlib` allows importing modules by name at runtime. `platform` gives OS and platform details.

3. `STRICT = False`
   - A boolean flag. When False the cell only reports problems. When True it runs extra checks and fails on missing or out of date packages.

4. `CORE = ["pandas", "numpy", "sklearn", "tensorflow"]`
   - A list of packages considered required for core notebook functionality.

5. `OPTIONAL = ["torch", "ultralytics", "transformers", "sentence_transformers"]`
   - Packages that are useful for extra experiments but not required for the main flow.

6. `def get_ver(mod, attr="__version__"):`
   - Start of a helper function that tries to import a module and read its version attribute.

7. `    try:`
   - Begin a safe import attempt.

8. `        m = importlib.import_module(mod)`
   - Import the module specified by the string `mod`.

9. `        return getattr(m, attr, "0.0.0")`
   - Return the value of the requested attribute if present, otherwise return the string "0.0.0" as a fallback.

10. `    except Exception:`
    - If the import fails for any reason, handle it here.

11. `        return None`
    - Return None to indicate the module is not installed or importable.

12. `def vtuple(v):`
    - Define a helper that converts a version string into a comparable Version object.

13. `    from packaging.version import Version`
    - Import the `Version` class locally inside the function to avoid a global dependency unless needed.

14. `    return Version(v or "0")`
    - Convert `v` into a `Version` object. If `v` is falsy, use "0".

15. `problems, warnings = [], []`
    - Prepare two lists. `problems` collects blocking issues. `warnings` collects notes that do not stop the run.

16. `installed = {m: get_ver(m) for m in CORE + OPTIONAL}`
    - Build a dictionary mapping each package name to the version string or None if not installed.

17. `#Core presence check `
    - A comment marking the start of required package presence checks.

18. `for m in CORE:`
    - Iterate over each package in the core list.

19. `    if installed[m] is None:`
    - If the package was not found (get_ver returned None), mark it as a blocking problem.

20. `        problems.append(f"Missing required package: {m} (pip install {m.replace('_','-')})")`
    - Append a helpful message that tells the user how to install the missing package.

21. `# Known-breaking combo guard`
    - A comment indicating checks for incompatible package combinations will follow.

22. `st = installed["sentence_transformers"]` and `tr = installed["transformers"]`
    - Read the recorded versions for two packages that have known compatibility constraints.

23. `if st is not None and tr is not None:`
    - Continue only if both packages are installed.

24. `    if vtuple(st) < vtuple("3.0.0") and vtuple(tr) >= vtuple("4.48.0"):`
    - If `sentence_transformers` is older than 3.0.0 and `transformers` is 4.48.0 or newer, treat this as a blocking incompatibility.

25. `        problems.append("Incompatible pair: sentence-transformers<3.0 with transformers>=4.48.\nFix: pip install 'sentence-transformers>=3.0.0' OR pin 'transformers<4.48'.")`
    - Add an explanatory problem message with a suggested fix.

26. `# TensorFlow vs NumPy heads-up `
    - A comment indicating a known source of runtime warnings.

27. `tf = installed["tensorflow"]` and `npv = installed["numpy"]`
    - Read the versions for TensorFlow and NumPy from the dictionary.

28. `if tf is not None and npv is not None and vtuple(npv) >= vtuple("2.0.0"):`
    - If both packages are present and NumPy is 2.0 or newer, add a non-blocking warning. Some TensorFlow builds may be incompatible with NumPy 2.x.

29. `    warnings.append(...)`
    - The message explains the potential issue and notes that if TensorFlow imports fine, the user can ignore it.

30. `print("Python:", sys.version.split()[0], "| Platform:", platform.platform())`
    - Print the Python version and platform for debugging.

31. `print("Detected versions:", {k: v for k, v in installed.items() if v})`
    - Print the versions that were found so the user can see the environment snapshot.

32. `if problems:`
    - If any blocking problems were detected, print them and exit.

33. `    raise SystemExit(1)`
    - Stop execution with a nonzero exit code so subsequent cells do not run in a broken environment.

34. `if warnings:`
    - If there are non-blocking warnings, print them for visibility.

35. `if STRICT:` and the strict checks block
    - If strict mode is enabled, check for minimum versions for a set of packages and fail if the requirements are not met.

36. `print("\nPreflight OK — proceeding.")`
    - If there were no blocking problems, show a confirmation message.

Inputs and outputs

- Inputs: environment, installed packages available in the Python interpreter.
- Outputs: printed environment summary, warnings, and possibly immediate Notebook exit on critical problems.

Notes and tips

- Running this cell first helps to avoid confusing runtime errors later.
- If you plan to run the notebook on a new machine, install packages from `requirements.txt` and re-run this cell first.
- Toggle `STRICT = True` if you want the notebook to enforce tighter version constraints.


---

End of 01 Preflight check explanation
