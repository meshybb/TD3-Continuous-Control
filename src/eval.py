import argparse
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot TD3 results")
    parser.add_argument("--env", required=True, help="Environment name (e.g., HalfCheetah-v4)")
    parser.add_argument("--results_dir", default="results", help="Directory containing the JSON files")
    args = parser.parse_args()

    # Find all JSON files for this environment
    search_pattern = os.path.join(args.results_dir, f"{args.env}_seed*_results.json")
    result_files = glob.glob(search_pattern)

    if not result_files:
        print(f"No result files found matching: {search_pattern}")
        return

    all_rewards = []
    finals = []
    eval_steps = None

    print(f"--- Analyzing {len(result_files)} runs for {args.env} ---")

    colors = ["royalblue", "darkorange", "forestgreen", "purple", "brown"]

    for i, file_path in enumerate(result_files):
        with open(file_path, "r") as f:
            data = json.load(f)
            
        seed = data["seed"]
        rewards = data["eval_rewards"]
        final_r = data["final_reward"]
        
        if eval_steps is None:
            eval_steps = data["eval_steps"]
            
        all_rewards.append(rewards)
        finals.append(final_r)
        
        print(f"Seed {seed}: Final Reward = {final_r:.2f}")

        # --- יצירת גרף נפרד לכל ריצה (בדיוק מה שהמרצה ביקש) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        color = colors[i % len(colors)]
        ax.plot(eval_steps, rewards, color=color, linewidth=2.0)
        ax.set_xlabel("Training Steps", fontsize=13)
        ax.set_ylabel("Evaluation Reward", fontsize=13)
        ax.set_title(f"TD3 — {args.env}   |   Run {i+1} (Seed {seed})", fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # סימון קו אדום של התוצאה הסופית
        ax.axhline(final_r, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Final: {final_r:.1f}")
        ax.legend(fontsize=11)
        fig.tight_layout()
        
        ind_plot_path = os.path.join(args.results_dir, f"{args.env}_run{i+1}_seed{seed}_learning_curve.png")
        fig.savefig(ind_plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved individual plot → {ind_plot_path}")

    # Calculate statistics
    mean_final = np.mean(finals)
    std_final = np.std(finals)
    
    print("\n--- Summary ---")
    print(f"Final rewards: {[float(f'{r:.2f}') for r in finals]}")
    print(f"Mean ± Std   : {mean_final:.2f} ± {std_final:.2f}")

    # --- יצירת גרף משולב ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rewards in enumerate(all_rewards):
        ax.plot(eval_steps, rewards, color=colors[i % len(colors)],
                linewidth=1.5, label=f"Run {i+1}", alpha=0.7)

    # Plot mean curve
    mean_rewards = np.mean(all_rewards, axis=0)
    ax.plot(eval_steps, mean_rewards, color="crimson", linewidth=2.5,
            linestyle="--", label="Mean")

    ax.set_xlabel("Training Steps",    fontsize=13)
    ax.set_ylabel("Evaluation Reward", fontsize=13)
    ax.set_title(f"TD3 — {args.env}",  fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    comb_plot_path = os.path.join(args.results_dir, f"{args.env}_combined_learning_curves.png")
    fig.savefig(comb_plot_path, dpi=150)
    plt.close(fig)
    print(f"\nCombined plot saved to → {comb_plot_path}")

if __name__ == "__main__":
    main()