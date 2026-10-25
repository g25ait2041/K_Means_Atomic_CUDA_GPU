"""Plot measured GPU benchmark results (benchmark.py CSV) -> 3-panel figure.
Usage: python plot_results.py results/benchmark_T4_colab.csv figures/fig5_gpu_benchmark.png"""
import sys, csv, collections, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1])))
tot, upd = collections.defaultdict(list), collections.defaultdict(list)
for r in rows:
    k = (r["mode"], int(r["N"]), int(r["D"]), int(r["K"])); it = int(r["iters"])
    tot[k].append(float(r["time_ms"]) / it)
    if r["mode"] != "cpu": upd[k].append(float(r["update_ms"]) / it)
mean = lambda d, k: sum(d[k]) / len(d[k])
gpu = next((r["gpu"] for r in rows if r.get("gpu")), "GPU")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
D, K = 8, 16
for mode, m in (("atomic", "o"), ("reduction", "s"), ("cpu", "^")):
    pts = sorted((n, mean(tot, k)) for k in tot if k[0] == mode and k[2] == D and k[3] == K for n in [k[1]])
    if pts: ax[0].loglog(*zip(*pts), m + "-", label=mode if mode != "cpu" else "CPU (1 core)")
ax[0].set_xlabel("N"); ax[0].set_ylabel("ms / iteration (total)"); ax[0].set_title(f"Total time vs N  (D={D}, K={K})"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
N = max(k[1] for k in tot)
for d, c in ((2, "C0"), (8, "C1"), (32, "C2")):
    for mode, ls, m in (("atomic", "--", "o"), ("reduction", "-", "s")):
        pts = sorted((k[3], mean(upd, k)) for k in upd if k[0] == mode and k[1] == N and k[2] == d)
        if pts: ax[1].loglog(*zip(*pts), ls, marker=m, color=c, label=f"{mode}, D={d}")
ax[1].set_xlabel("K"); ax[1].set_ylabel("update step ms / iteration"); ax[1].set_title(f"Centroid-update cost vs K  (N={N:,})"); ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=.3, which="both")
from matplotlib.ticker import NullFormatter, FixedLocator
ax[1].xaxis.set_major_locator(FixedLocator([4, 16, 64])); ax[1].set_xticklabels(["4", "16", "64"]); ax[1].xaxis.set_minor_formatter(NullFormatter())
N1 = 1_000_000; labels, sp_a, sp_r = [], [], []
for k in sorted({(k[2], k[3]) for k in tot if k[1] == N1}):
    c = mean(tot, ("cpu", N1, *k)); labels.append(f"D={k[0]}\nK={k[1]}")
    sp_a.append(c / mean(tot, ("atomic", N1, *k))); sp_r.append(c / mean(tot, ("reduction", N1, *k)))
x = range(len(labels)); ax[2].bar([i - 0.2 for i in x], sp_a, 0.4, label="atomic"); ax[2].bar([i + 0.2 for i in x], sp_r, 0.4, label="reduction")
ax[2].set_xticks(list(x)); ax[2].set_xticklabels(labels, fontsize=8); ax[2].set_ylabel("speed-up vs 1-core CPU (×)"); ax[2].set_title(f"Speed-up at N = {N1:,}"); ax[2].legend(); ax[2].grid(alpha=.3, axis="y")
fig.suptitle(f"Measured on {gpu} (Google Colab), 50 iterations, mean of 2 runs", fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=160); print("wrote", sys.argv[2])
