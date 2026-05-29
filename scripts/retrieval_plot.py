import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----- 1. Data (4B-scale models + EmbeddingGemma) -----
data = {
    "Model": [
        "text-embedding-3-large\n(OpenAI)",
        "pplx-embed-v1-4b\n(Perplexity)",
        "Qwen3-Embedding-4B\n(Qwen)",
        "embeddinggemma-300m\n(Google)",
        "Fine-tuned XLM-R-Large\n(Factiverse)"
    ],
    "F1": [0.7959, 0.8177, 0.7500, 0.5341, 0.8216]
}

df = pd.DataFrame(data)

# ----- 2. Sort by F1 -----
df.sort_values("F1", ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# ----- 3. One color per method -----
color_map = {
    "text-embedding-3-large\n(OpenAI)":      "#E07B39",
    "pplx-embed-v1-4b\n(Perplexity)":        "#6A5ACD",
    "Qwen3-Embedding-4B\n(Qwen)":            "#2ECC71",
    "embeddinggemma-300m\n(Google)":         "#EA4335",
    "Fine-tuned XLM-R-Large\n(Factiverse)":   "#2E86C1"
}
colors = [color_map[m] for m in df["Model"]]

# ----- 4. Plot -----
fig, ax = plt.subplots(figsize=(10, 5.5))

bars = ax.barh(df["Model"], df["F1"], color=colors, height=0.55, edgecolor="white")

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}",
        ha="left", va="center", fontsize=12, fontweight="bold"
    )

# Highlight best model with gold border
best_idx = df["F1"].idxmax()
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(2.5)

# ----- 5. Formatting -----
ax.set_xlim(0, 0.95)
ax.set_xlabel("F1 Score", fontsize=12)
ax.set_title("Evidence Retrieval — F1 Score by Model", fontsize=14, fontweight="bold", pad=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
fig.savefig("evidence_retrieval_f1_with_gemma.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: evidence_retrieval_f1_with_gemma.png")