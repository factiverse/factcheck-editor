import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----- 1. Load Data -----
df = pd.read_csv("data/claim_detection/f1_scores.tsv", sep="\t")
# ----- 2. Compute Mean Macro F1 per Model -----
mean_scores = {
    "Qwen3-8b": df["qwen3_macro"].mean(),
    "Claude-Opus-4.6": df["claude_opus_4_6_macro"].mean(),
    "GPT-5.2": df["gpt52_macro"].mean(),
    "Factiverse\n(Fine-tuned XLM-R)": df["facti_macro"].mean()
}

models = list(mean_scores.keys())
scores = list(mean_scores.values())

# ----- 3. Use same colors from your original code -----
colors = ["green", "gold", "red", "blue"]

# ----- 4. Plot -----
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(models, scores, color=colors, width=0.55, 
              edgecolor="white", linewidth=1.2)

# Add value labels on top of each bar
for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.3f}",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )

ax.set_ylim(0, 1.0)
ax.set_ylabel("Mean Macro F1 Score", fontsize=12)
ax.set_title(
    "Claim Detection — Average Macro F1 Across All Languages",
    fontsize=14, fontweight="bold", pad=14
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
fig.savefig("claim_detection_mean_f1.png", dpi=300, bbox_inches="tight")
plt.show()