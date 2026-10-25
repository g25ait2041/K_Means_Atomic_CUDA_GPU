"""Binary I/O shared by all Python tools.  Format: int32 N, int32 D, float32[N*D]."""
import numpy as np

def write_points(path, X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(X.shape, dtype=np.int32).tofile(f)
        X.tofile(f)

def read_points(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)

def read_assign(path):
    with open(path, "rb") as f:
        n = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        return np.fromfile(f, dtype=np.int32, count=n)
