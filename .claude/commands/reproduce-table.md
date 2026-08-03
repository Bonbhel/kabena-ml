---
description: Run a validation table and compare against the paper's published values
---
Argument: the table number (1-5).

1. Confirm the environment first:
   python -c "import sys, kabena; print(sys.prefix, kabena.__version__)"
   Expect the repo .venv and 2.1.1. If not, STOP.
2. Run validation/table<N>_*.py
3. Compare every figure against the values printed by the script itself as the
   paper reference (each script prints them).
4. Report a short table: metric | obtained | paper | delta | verdict.
5. Flag anything outside +/- 0.005 as a REGRESSION and stop there.

Do not "fix" a discrepancy on your own — report it and wait.
