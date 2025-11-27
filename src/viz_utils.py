from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick
import textwrap
from src.clean_utils import is_non_product, clean_non_products
import numpy as np

def shorten_labels(index, max_len=32):
    """Shorten overly long labels for plotting."""
    return [textwrap.shorten(str(x), width=max_len, placeholder="…") for x in index]

def plot_top_copurchase_horizontal(series, title, outpath, theme="zabka", top_n=15, label_max_len=32):
    # --- theme config ---
    themes = {
        "zabka":    ("#39A935", "white", "#333333"),
        "business": ("#1F77B4", "white", "#333333"),
        "dark":     ("#4CAF50", "#111111", "#FAFAFA"),
    }
    bar_color, bg, txt = themes.get(theme, themes["zabka"])

    # clean & reduce
    series = clean_non_products(series)
    series = series.sort_values(ascending=False).head(top_n)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # sort ascending for horizontal bars (small at top, big at bottom)
    s = series.sort_values()
    y_labels = shorten_labels(s.index, max_len=label_max_len)

    ax.barh(y_labels, s.values, color=bar_color, edgecolor="white", linewidth=1.0)

    ax.set_title(title, fontsize=18, color=txt, pad=18)
    ax.set_xlabel("Co-purchase quantity (line count)", fontsize=14, color=txt)
    ax.tick_params(axis="y", labelsize=11, labelcolor=txt)
    ax.tick_params(axis="x", labelcolor=txt)

    # value labels on bars
    max_val = s.values.max() if len(s) else 0
    for i, v in enumerate(s.values):
        ax.text(
            v + max_val * 0.01,
            i,
            str(int(v)),
            va="center",
            fontsize=11,
            color=txt,
        )

    ax.grid(axis="x", linestyle="--", alpha=0.3, color=txt)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, facecolor=bg)
    plt.close(fig)

def save_bar(series: pd.Series, title: str, outpath: str):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    if s.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        s.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def save_hist(series: pd.Series, bins: int, title: str, outpath: str, log=False):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    ax = series.plot(kind="hist", bins=bins, log=log)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_box(series: pd.Series, title: str, outpath: str, log=False):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.boxplot(series.dropna(), vert=True, showfliers=True)
    ax.set_title(title)
    if log: ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_basket_fc_by_slot(series, title, outpath):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        series.index,
        series.values,
        color="#2A9D8F",       # modern teal/green accent
        edgecolor="white",
        linewidth=1.2,
        width=0.65
    )

    # Title & labels
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Share of baskets", fontsize=14)

    # Y-axis as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Grid (clean, subtle)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X-tick style
    plt.xticks(
        rotation=22,
        ha="right",
        fontsize=12
    )

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),      # offset above bar
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    
def plot_basket_fc_by_slot_thematic(series, title, outpath, theme="dark"):
    """
    series: index = slot_label, values = share (0–1)
    theme: 'zabka', 'business', 'dark'
    """
    # --- themes ---
    themes = {
        "zabka": {
            "bg": "white",
            "axes_bg": "white",
            "bar": "#39A935",          # Żabka green-ish
            "edge": "white",
            "grid": "#CCCCCC",
            "title": "#222222",
            "label": "#333333",
        },
        "business": {
            "bg": "white",
            "axes_bg": "white",
            "bar": "#1F77B4",          # classic business blue
            "edge": "white",
            "grid": "#D0D0D0",
            "title": "#111111",
            "label": "#333333",
        },
        "dark": {
            "bg": "#111111",
            "axes_bg": "#111111",
            "bar": "#4CAF50",          # bright green on dark
            "edge": "#222222",
            "grid": "#444444",
            "title": "#FAFAFA",
            "label": "#E0E0E0",
        },
    }

    cfg = themes.get(theme, themes["zabka"])

    # --- figure ---
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(cfg["bg"])
    ax.set_facecolor(cfg["axes_bg"])

    series = series[series.index != "Probably outliers"]
    
    bars = ax.bar(
        series.index,
        series.values,
        color=cfg["bar"],
        edgecolor=cfg["edge"],
        linewidth=1.2,
        width=0.65,
    )

    # labels & title
    ax.set_title(title, fontsize=18, pad=20, color=cfg["title"])
    ax.set_xlabel("")
    ax.set_ylabel("Share of baskets", fontsize=14, color=cfg["label"])

    # y-axis = percent
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelcolor=cfg["label"])
    ax.tick_params(axis="x", labelcolor=cfg["label"])

    # grid
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6, color=cfg["grid"])

    # remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if theme == "dark":
        # make remaining spines subtle but visible
        ax.spines["left"].set_color(cfg["grid"])
        ax.spines["bottom"].set_color(cfg["grid"])

    # x-ticks
    plt.xticks(rotation=22, ha="right", fontsize=12)

    # annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            color=cfg["label"],
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, facecolor=cfg["bg"])
    plt.close(fig)
    

def plot_fc_heatmap_weekday_slot(fc_heat, outpath, theme="zabka"):
    """
    fc_heat: DataFrame, index=weekday, columns=slot_label, values = FC line counts
    """
    # --- themes ---
    themes = {
        "zabka":    ("white", "Greens",  "#333333"),
        "business": ("white", "Blues",   "#333333"),
        "dark":     ("#111111", "Greens", "#FAFAFA"),
    }
    bg, cmap_name, txt = themes.get(theme, themes["zabka"])

    data = fc_heat.values
    rows, cols = data.shape

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Heatmap
    im = ax.imshow(data, aspect="auto", cmap=cmap_name)

    # Ticks & labels
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(fc_heat.index, fontsize=11, color=txt)

    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(fc_heat.columns, fontsize=11, rotation=30, ha="right", color=txt)

    ax.set_xlabel("Time slot", fontsize=13, color=txt)
    ax.set_ylabel("Weekday", fontsize=13, color=txt)
    ax.set_title("Food Corner line count — weekday × time slot", fontsize=16, color=txt, pad=14)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=txt)
    for label in cbar.ax.get_yticklabels():
        label.set_color(txt)
    cbar.ax.set_ylabel("FC line count", rotation=-90, va="bottom", color=txt)

    # Optional: annotate cells with counts (only if not too huge)
    max_val = data.max()
    for i in range(rows):
        for j in range(cols):
            v = int(data[i, j])
            if v == 0:
                continue
            ax.text(
                j, i, str(v),
                ha="center", va="center",
                fontsize=9,
                color="white" if data[i, j] > max_val * 0.5 else txt,
            )

    # Clean grid / spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, facecolor=bg)
    plt.close(fig)
    
def plot_top_fc_anchors(series, outpath, theme="dark"):
    themes = {
        "zabka":    ("#39A935", "white", "#333333"),
        "business": ("#1F77B4", "white", "#333333"),
        "dark":     ("#4CAF50", "#111111", "#FAFAFA"),
    }
    bar_color, bg, txt = themes.get(theme, themes["zabka"])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    s = series.sort_values()  # smallest at top, biggest at bottom
    ax.barh(s.index, s.values, color=bar_color, edgecolor="white", linewidth=1.0)

    ax.set_title("Top Food Corner items (by baskets with FC)", fontsize=18, color=txt, pad=16)
    ax.set_xlabel("Number of baskets", fontsize=14, color=txt)
    ax.tick_params(axis="y", labelsize=11, labelcolor=txt)
    ax.tick_params(axis="x", labelcolor=txt)

    # Labels on bars
    for i, v in enumerate(s.values):
        ax.text(
            v + max(s.values)*0.01,
            i,
            str(v),
            va="center",
            fontsize=11,
            color=txt,
        )

    ax.grid(axis="x", linestyle="--", alpha=0.3, color=txt)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, facecolor=bg)
    plt.close(fig)

def plot_fc_copurchase_tilemap(mat, outpath, theme="dark"):
    themes = {
        "zabka":    ("white", "#39A935", "#333333"),
        "business": ("white", "#1F77B4", "#333333"),
        "dark":     ("#63817F", "#4CAF50", "#030303"),
    }
    bg, main, txt = themes.get(theme, themes["zabka"])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Convert to numpy
    data = mat.values

    # Colour map: light→dark main color
    cmap = plt.cm.get_cmap("Greens" if theme == "zabka" else
                           "Blues"   if theme == "business" else
                           "Greens").copy()
    # Heatmap
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    # Ticks & labels
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=10, color=txt)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, fontsize=9, rotation=45, ha="right", color=txt)

    ax.set_xlabel("Co-purchased product", fontsize=14, color=txt)
    ax.set_ylabel("Food Corner anchor", fontsize=14, color=txt)
    ax.set_title("What do people buy with each top FC item?", fontsize=18, color=txt, pad=16)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=txt)
    plt.setp(cbar.ax.get_yticklabels(), color=txt)

    norm = im.norm  # normalization function used by the colormap
    cmap_obj = im.cmap

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = data[i, j]
            if val > 0:
                ax.text(
                    j, i, int(val),
                    ha="center", va="center",
                    fontsize=8, color=txt
                )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, facecolor=bg)
    plt.close(fig)