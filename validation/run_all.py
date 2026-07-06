"""Exécute les trois tables du preprint. QUICK=1 pour un smoke test rapide."""
import subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
for script in ("table1_real_datasets.py", "table2_extreme_imbalance.py", "table3_moments_noise.py", "table4_rivals.py", "table5_sensitivity.py"):
    print("\n" + "="*60)
    r = subprocess.run([sys.executable, os.path.join(here, script)], cwd=here)
    if r.returncode != 0:
        sys.exit(r.returncode)
print("\nToutes les validations sont passées.")
