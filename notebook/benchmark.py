"""Benchmark sweep over N, D, K for atomic vs reduction (and CPU baseline).
Parses the RESULT line printed by the executables."""
import argparse, subprocess, re, csv, os, tempfile, numpy as np
from kmeans_io import write_points

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--cpu", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--sizes", default="100000,1000000,4000000")
ap.add_argument("--dims", default="2,8,32"); ap.add_argument("--ks", default="4,16,64")
ap.add_argument("--repeat", type=int, default=3); ap.add_argument("--cpu-max-n", type=int, default=1000000)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
rows = []
def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    m = re.search(r"RESULT (.*)", out); kv = dict(p.split("=", 1) for p in re.findall(r'(\w+=(?:"[^"]*"|\S+))', m.group(1)))
    return {k: v.strip('"') for k, v in kv.items()}
rng = np.random.default_rng(0)
for N in map(int, a.sizes.split(",")):
    for D in map(int, a.dims.split(",")):
        for K in map(int, a.ks.split(",")):
            tmp = tempfile.gettempdir(); path = os.path.join(tmp, f"bench_{N}_{D}_{K}.bin")
            ta, tc = os.path.join(tmp, "a.bin"), os.path.join(tmp, "c.bin")
            centers = rng.uniform(-10, 10, (K, D)).astype(np.float32)
            X = centers[rng.integers(0, K, N)] + rng.normal(0, 1, (N, D)).astype(np.float32)
            write_points(path, X)
            for mode in ("atomic", "reduction"):
                for r in range(a.repeat):
                    res = run([a.exe, "--data", path, "--k", str(K), "--mode", mode, "--iters", "50",
                               "--out-assign", ta, "--out-cent", tc])
                    res["rep"] = r; rows.append(res); print(res)
            if a.cpu and N <= a.cpu_max_n:
                res = run([a.cpu, "--data", path, "--k", str(K), "--iters", "50",
                           "--out-assign", ta, "--out-cent", tc])
                res["rep"] = 0; rows.append(res); print(res)
            os.remove(path)
keys = sorted({k for r in rows for k in r})
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", a.out)
