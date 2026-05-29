import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ----- 1. Load Data -----
# df = pd.read_csv("data/claim_detection/f1_scores.tsv")  # adjust path/separator as needed
# If tab-separated:
df = pd.read_csv("data/claim_detection/f1_scores.tsv", sep="\t")

# ----- 2. Extract only Macro F1 columns -----
macro_cols = {
    "claude_opus_4_6_macro": "Claude Opus 4.6",
    "gpt52_macro": "GPT 5.2",
    "facti_macro": "Factiverse (XLM-R)",
    "qwen3_macro": "Qwen3-8b"
}

heatmap_df = df[["lang"] + list(macro_cols.keys())].copy()
heatmap_df.rename(columns=macro_cols, inplace=True)
heatmap_df.set_index("lang", inplace=True)

# ----- 3. Sort languages by average F1 (best at top) -----
heatmap_df["avg"] = heatmap_df.mean(axis=1)
heatmap_df.sort_values("avg", ascending=False, inplace=True)
heatmap_df.drop(columns="avg", inplace=True)

# ----- 4. Create the Heatmap -----
fig, ax = plt.subplots(figsize=(8, 28))  # tall figure for many languages

sns.heatmap(
    heatmap_df,
    annot=True,          # show F1 values in cells
    fmt=".2f",           # two decimal places
    cmap="RdYlGn",       # red = low, green = high
    vmin=0.0,
    vmax=1.0,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Macro F1 Score", "shrink": 0.5},
    ax=ax
)

ax.set_title("Claim Detection — Macro F1 by Language & Model", fontsize=14, pad=12)
ax.set_ylabel("Language", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.tick_params(axis="y", labelsize=7)
ax.tick_params(axis="x", labelsize=10, rotation=15)

plt.tight_layout()

# ----- 5. Save as high-res image for PowerPoint -----
fig.savefig("claim_detection_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: claim_detection_heatmap.png — drag this into your PowerPoint slide.")