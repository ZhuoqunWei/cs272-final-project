# plots/make_plots.py
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def smooth(y, w=50):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode="valid")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor_csv", required=True)  # from Monitor()
    ap.add_argument("--eval_csv", required=True)     # from eval_500.py
    ap.add_argument("--out_dir", default="plots_out")
    ap.add_argument("--prefix", default="13_emergency_yield")  # 13 = learning curve; 14 = violin
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # --- Learning curve (ID 13) ---
    dfm = pd.read_csv(args.monitor_csv, comment="#")
    ep = np.arange(1, len(dfm["r"])+1)
    sm = smooth(dfm["r"].values, w=50)
    plt.figure()
    plt.plot(ep[:len(sm)], sm)
    plt.xlabel("Training Episodes"); plt.ylabel("Mean Episodic Return (smoothed)")
    plt.title("Learning Curve — emergency-yield-v0")
    plt.tight_layout()
    lc_path = out / f"{args.prefix}_learning_curve.png"  # ID 13
    plt.savefig(lc_path, dpi=180); plt.close()

    # --- Violin (ID 14) ---
    dfe = pd.read_csv(args.eval_csv)
    plt.figure()
    plt.violinplot(dfe["return"].values, showmeans=True)
    plt.ylabel("Episodic Return (no exploration)")
    plt.title("Performance — 500 Episodes — emergency-yield-v0")
    plt.tight_layout()
    vp_path = out / f"14_emergency_yield_performance_violin.png"  # ID 14
    plt.savefig(vp_path, dpi=180); plt.close()

    # (optional) quick success/time_to_clear text log
    succ_rate = dfe["success"].mean()*100 if "success" in dfe else float("nan")
    print(f"Wrote {lc_path} and {vp_path} | success rate: {succ_rate:.1f}%")
