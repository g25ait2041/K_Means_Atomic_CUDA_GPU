"""Verify GPU outputs against the CPU/NumPy reference."""
import sys, numpy as np
from kmeans_io import read_points, read_assign

def inertia(X, C, A): return float(((X - C[A]) ** 2).sum())

X = read_points(sys.argv[1]); Cr = read_points(sys.argv[2]); Ar = read_assign(sys.argv[3])
print(f"reference: inertia={inertia(X, Cr, Ar):.6e}")
ok = True
for i in range(4, len(sys.argv), 2):
    C = read_points(sys.argv[i]); A = read_assign(sys.argv[i + 1])
    agree = 100.0 * (A == Ar).mean(); err = float(np.abs(C - Cr).max())
    inr = inertia(X, C, A); rel = abs(inr - inertia(X, Cr, Ar)) / inertia(X, Cr, Ar)
    status = "PASS" if (agree > 99.9 and err < 1e-3) else "CHECK"
    if status != "PASS": ok = False
    print(f"{sys.argv[i]:>28s}: agreement={agree:7.3f}%  centroid_max_abs_err={err:.2e}  inertia={inr:.6e} (rel diff {rel:.1e})  {status}")
sys.exit(0 if ok else 1)
