"""Vectorised NumPy K-Means reference (identical init + stopping rule to the CUDA code).
Returns assignments, centroids, iterations, per-iteration history."""
import numpy as np

def kmeans_numpy(X, K, max_iters=100, tol=1e-4, init=None, history=False):
    X = X.astype(np.float32)
    C = (X[:K] if init is None else init).astype(np.float32).copy()
    A = np.full(len(X), -1, dtype=np.int32)
    hist = []
    for it in range(1, max_iters + 1):
        # squared distances via ||x||^2 - 2 x.c + ||c||^2  (N x K), argmin over K
        d2 = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
        newA = d2.argmin(1).astype(np.int32)
        changed = int((newA != A).sum()); A = newA
        newC = C.copy(); max_shift2 = 0.0
        for k in range(K):
            m = A == k
            if m.any():
                newC[k] = X[m].mean(0)
                max_shift2 = max(max_shift2, float(((newC[k] - C[k]) ** 2).sum()))
        C = newC
        if history:
            inertia = float(((X - C[A]) ** 2).sum()); hist.append((it, changed, np.sqrt(max_shift2), inertia))
        if changed == 0 or max_shift2 < tol * tol:
            break
    return A, C, it, hist
