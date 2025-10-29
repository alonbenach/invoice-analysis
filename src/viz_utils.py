from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

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
