"""
Generate individual learning curve plots — one per training run (seed).

Outputs (6 files total):
  results/InvertedPendulum-v4_run1_seed42_learning_curve.png
  results/InvertedPendulum-v4_run2_seed55_learning_curve.png
  results/InvertedPendulum-v4_run3_seed68_learning_curve.png
  results/HalfCheetah-v4_run1_seed42_learning_curve.png
  results/HalfCheetah-v4_run2_seed55_learning_curve.png
  results/HalfCheetah-v4_run3_seed68_learning_curve.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
SEEDS       = [42, 55, 68]
COLORS      = ["royalblue", "darkorange", "forestgreen"]


def save_plot(env, run_num, seed, steps, rewards, color, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, rewards, color=color, linewidth=2.0)
    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Evaluation Reward", fontsize=13)
    ax.set_title(f"TD3 — {env}   |   Run {run_num} (Seed {seed})", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(rewards[-1], color="crimson", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Final: {rewards[-1]:.1f}")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── InvertedPendulum ──────────────────────────────────────────────────────────
print("InvertedPendulum-v4")
with open(os.path.join(RESULTS_DIR, "InvertedPendulum-v4_results.json")) as f:
    ip_data = json.load(f)

for i, (seed, rewards) in enumerate(zip(ip_data["seeds"], ip_data["all_rewards"])):
    out = os.path.join(RESULTS_DIR,
                       f"InvertedPendulum-v4_run{i+1}_seed{seed}_learning_curve.png")
    save_plot("InvertedPendulum-v4", i + 1, seed,
              ip_data["all_steps"], rewards, COLORS[i], out)

# ── HalfCheetah ───────────────────────────────────────────────────────────────
print("HalfCheetah-v4")
for i, seed in enumerate(SEEDS):
    json_path = os.path.join(RESULTS_DIR, f"HalfCheetah-v4_seed{seed}_results.json")
    with open(json_path) as f:
        hc_data = json.load(f)
    out = os.path.join(RESULTS_DIR,
                       f"HalfCheetah-v4_run{i+1}_seed{seed}_learning_curve.png")
    save_plot("HalfCheetah-v4", i + 1, seed,
              hc_data["eval_steps"], hc_data["eval_rewards"], COLORS[i], out)

print("Done.")
