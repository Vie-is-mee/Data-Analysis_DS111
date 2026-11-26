"""M2-13 · Hình phân tích 300 DPI cho slide và báo cáo.

Quy ước: tiêu đề nêu mã câu hỏi và nội dung; phụ đề nêu phương pháp, n và
phiên bản dữ liệu. Dùng ``matplotlib.figure.Figure`` trực tiếp (không dùng
pyplot) để chạy được trên máy không có màn hình và không giữ trạng thái chung.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.analysis.eda_summary import vn_number
from src.analysis.group_analysis import FOCUS_CLUSTERS

DPI = 300
BASE_FONT = 11
PALETTE: tuple[str, ...] = ("#1f4e79", "#c55a11", "#548235", "#7f7f7f", "#7030a0")
CLUSTER_COLORS: dict[str, str] = {"JP": "#1b9e77", "US": "#d95f02", "DE": "#7570b3", "other": "#7f7f7f"}
SPEC_LABELS: dict[str, str] = {
    "none": "Không kiểm soát",
    "odometer": "Cố định số dặm",
    "odometer_type": "Cố định số dặm + loại xe",
}
CONDITION_ORDER: tuple[str, ...] = ("salvage", "fair", "good", "excellent", "like new", "new", "unknown")


def new_figure(width: float = 10.0, height: float = 5.4, *, ncols: int = 1, sharey: bool = False):
    """Tạo Figure và mảng trục đã định dạng thống nhất."""
    fig = Figure(figsize=(width, height))
    axes = fig.subplots(1, ncols, sharey=sharey, squeeze=False)[0]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=BASE_FONT - 1)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    return fig, axes


def save_figure(fig: Figure, out_dir: Path, name: str, *, title: str, subtitle: str) -> Path:
    """Gắn tiêu đề/phụ đề và lưu PNG 300 DPI; trả về đường dẫn."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=BASE_FONT + 3, fontweight="bold")
    fig.text(0.01, 0.01, subtitle, fontsize=BASE_FONT - 2, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = Path(out_dir) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    return path


def subtitle(method: str, n: float, version: str | None) -> str:
    return f"{method} · n = {vn_number(n)} · dữ liệu: {version or 'chưa ghi phiên bản'}"


def plot_price_distribution(df: pd.DataFrame, out_dir: Path, version: str | None) -> Path:
    """Q1 · Phân phối giá trên thang USD (≤ P99) và thang log."""
    price = df["price"].to_numpy(dtype=float)
    price = price[np.isfinite(price) & (price > 0)]
    fig, (ax_usd, ax_log) = new_figure(11, 4.8, ncols=2)
    cap = float(np.quantile(price, 0.99))
    ax_usd.hist(price[price <= cap], bins=60, color=PALETTE[0])
    ax_usd.set_xlabel("Giá (USD, ≤ P99)")
    ax_usd.set_ylabel("Số tin")
    ax_log.hist(price, bins=np.logspace(np.log10(price.min()), np.log10(price.max()), 60), color=PALETTE[1])
    ax_log.set_xscale("log")
    ax_log.set_xlabel("Giá (USD, trục log)")
    median = float(np.median(price))
    for ax in (ax_usd, ax_log):
        ax.axvline(median, color="black", linestyle="--", linewidth=1)
    return save_figure(
        fig, out_dir, "eda_Q1_price_distribution",
        title="Q1 · Phân phối giá: thang USD và thang log",
        subtitle=subtitle(f"Histogram; đường đứt = trung vị {vn_number(median)} USD", price.size, version),
    )


def plot_age_curve(curve: pd.DataFrame, out_dir: Path, version: str | None) -> Path:
    """Q3 · Trung vị và IQR của giá theo từng tuổi xe."""
    fig, (ax,) = new_figure(10, 5)
    ax.fill_between(curve["car_age"], curve["q1"], curve["q3"], color=PALETTE[0], alpha=0.15, label="Q1–Q3")
    ax.plot(curve["car_age"], curve["median"], color=PALETTE[0], linewidth=2, label="Trung vị")
    ax.set_yscale("log")
    ax.set_xlabel("Tuổi xe (năm)")
    ax.set_ylabel("Giá (USD, trục log)")
    lowest = curve.loc[curve["median"].idxmin()]
    ax.annotate(f"thấp nhất: {int(lowest['car_age'])} tuổi", xy=(lowest["car_age"], lowest["median"]),
                xytext=(10, 30), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    ax.legend(frameon=False)
    return save_figure(
        fig, out_dir, "eda_Q3_median_price_by_age",
        title="Q3 · Trung vị giá theo tuổi xe",
        subtitle=subtitle("Trung vị và IQR theo từng tuổi (chỉ tuổi có ≥ 30 tin)", curve["n"].sum(), version),
    )


def _ci_bar_values(table: pd.DataFrame, x: str, y: str, kind: str) -> tuple[float, float, float]:
    row = table[(table["x"] == x) & (table["y"] == y)]
    if row.empty:
        return (np.nan, np.nan, np.nan)
    prefix = "pearson" if kind == "pearson" else "spearman"
    value = row[f"{prefix}_r" if kind == "pearson" else "spearman_rho"].iloc[0]
    return (float(value), float(row[f"{prefix}_ci_low"].iloc[0]), float(row[f"{prefix}_ci_high"].iloc[0]))


def plot_correlation_comparison(
    table: pd.DataFrame, features: Sequence[str], out_dir: Path, version: str | None
) -> Path:
    """Q1 · Pearson với price, Pearson với log_price và Spearman (bất biến qua log)."""
    series = (("Pearson r với price", "price", "pearson"), ("Pearson r với log_price", "log_price", "pearson"),
              ("Spearman ρ (như nhau cho price và log)", "log_price", "spearman"))
    fig, (ax,) = new_figure(11, 5.2)
    positions, width = np.arange(len(features)), 0.27
    for i, (label, y, kind) in enumerate(series):
        values = np.array([_ci_bar_values(table, f, y, kind) for f in features], dtype=float)
        err = np.vstack([values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]])
        ax.bar(positions + (i - 1) * width, values[:, 0], width, yerr=err, capsize=2, color=PALETTE[i], label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(features))
    ax.set_ylabel("Hệ số tương quan")
    ax.legend(frameon=False, fontsize=BASE_FONT - 1)
    return save_figure(
        fig, out_dir, "eda_Q1_pearson_vs_spearman",
        title="Q1 · Pearson và Spearman của các biến số với giá",
        subtitle=subtitle("Thanh = KTC 95%; cylinders chỉ dùng giá trị quan sát", table["n_samples"].max(), version),
    )


CLUSTER_LABELS: dict[str, str] = {"JP": "Nhật (Toyota, Honda)", "US": "Mỹ (Ford, Chevrolet)",
                                  "DE": "Đức (BMW, Mercedes-Benz, Audi)"}


def plot_cluster_curves(curve: pd.DataFrame, out_dir: Path, version: str | None) -> Path:
    """Q2 · Đường trung vị giá theo tuổi của ba cụm hãng."""
    fig, (ax,) = new_figure(10, 5)
    for cluster in FOCUS_CLUSTERS:
        part = curve[curve["group_value"] == cluster]
        if part.empty:
            continue
        color = CLUSTER_COLORS[cluster]
        ax.plot(part["car_age"], part["median"], color=color, linewidth=2, label=CLUSTER_LABELS[cluster])
        ax.fill_between(part["car_age"], part["q1"], part["q3"], color=color, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Tuổi xe (năm)")
    ax.set_ylabel("Giá (USD, trục log)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Không có tuổi nào đủ 30 tin trong các cụm hãng", ha="center", transform=ax.transAxes)
    return save_figure(
        fig, out_dir, "eda_Q2_cluster_age_curves",
        title="Q2 · Trung vị giá theo tuổi của ba cụm hãng (xe 1–20 tuổi)",
        subtitle=subtitle("Trung vị và IQR theo từng tuổi (≥ 30 tin)", curve["n"].sum(), version),
    )


def plot_depreciation_forest(
    slopes: pd.DataFrame,
    groups: Sequence[str],
    out_dir: Path,
    version: str | None,
    *,
    name: str = "eda_Q2_depreciation_by_brand",
    title: str = "Q2 · Tốc độ mất giá theo hãng, có và không kiểm soát",
) -> Path:
    """Q2 · %/năm và %/10.000 dặm kèm KTC 95% (HC3) cho từng nhóm và cấu hình kiểm soát."""
    fig, (ax_year, ax_miles) = new_figure(12, 5.6, ncols=2, sharey=True)
    specs = [spec for spec in SPEC_LABELS if spec in set(slopes["controls"])]
    offsets = np.linspace(-0.25, 0.25, len(specs)) if len(specs) > 1 else np.zeros(1)
    y_pos = np.arange(len(groups))
    for offset, spec, color in zip(offsets, specs, PALETTE):
        part = slopes[slopes["controls"] == spec].set_index("group_value").reindex(list(groups))
        pct = part["pct_per_year"].to_numpy(dtype=float)
        err = np.vstack([pct - part["pct_ci_low"].to_numpy(dtype=float),
                         part["pct_ci_high"].to_numpy(dtype=float) - pct])
        ax_year.errorbar(pct, y_pos + offset, xerr=err, fmt="o", color=color, capsize=2, markersize=4,
                         label=SPEC_LABELS[spec])
        if spec != "none":
            miles = part["pct_per_10k_miles"].to_numpy(dtype=float)
            err_m = np.vstack([miles - part["pct_10k_ci_low"].to_numpy(dtype=float),
                               part["pct_10k_ci_high"].to_numpy(dtype=float) - miles])
            ax_miles.errorbar(miles, y_pos + offset, xerr=err_m, fmt="o", color=color, capsize=2, markersize=4)
    for ax in (ax_year, ax_miles):
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
    ax_year.set_yticks(y_pos)
    ax_year.set_yticklabels(list(groups))
    ax_year.invert_yaxis()
    ax_year.set_xlabel("% giá thay đổi mỗi năm tuổi")
    ax_miles.set_xlabel("% giá thay đổi mỗi 10.000 dặm")
    handles, labels = ax_year.get_legend_handles_labels()
    if handles:  # khung phải trống phía gần 0 nên đặt chú thích ở đó, không che điểm dữ liệu
        ax_miles.legend(handles, labels, frameon=False, fontsize=BASE_FONT - 2, loc="lower right")
    n_total = slopes.loc[slopes["controls"] == "none", "n"].sum()
    return save_figure(
        fig, out_dir, name, title=title,
        subtitle=subtitle("OLS ln(giá) ~ tuổi (+ kiểm soát), xe 1–20 tuổi; thanh = KTC 95% (HC3)", n_total, version),
    )


def plot_era_correlation(era_corr: pd.DataFrame, out_dir: Path, version: str | None) -> Path:
    """Q3 · Spearman ρ (tuổi xe, log giá) trong từng đời xe, kèm n."""
    order = ["Classic", "Old", "Mid", "Modern"]
    part = era_corr[(era_corr["x"] == "car_age") & (era_corr["y"] == "log_price")]
    part = part.set_index("group_value").reindex(order)
    rho = part["spearman_rho"].to_numpy(dtype=float)
    err = np.vstack([rho - part["spearman_ci_low"].to_numpy(dtype=float),
                     part["spearman_ci_high"].to_numpy(dtype=float) - rho])
    fig, (ax,) = new_figure(9, 5)
    colors = [PALETTE[1] if np.isfinite(v) and v > 0 else PALETTE[0] for v in rho]
    ax.bar(order, np.nan_to_num(rho), yerr=np.nan_to_num(err), capsize=3, color=colors)
    for i, (value, n) in enumerate(zip(rho, part["n_samples"].to_numpy(dtype=float))):
        if np.isfinite(value):
            ax.text(i, value + (0.03 if value >= 0 else -0.07), f"n = {vn_number(n)}", ha="center",
                    fontsize=BASE_FONT - 2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman ρ (tuổi xe, log giá)")
    return save_figure(
        fig, out_dir, "eda_Q3_rho_by_era",
        title="Q3 · Spearman ρ giữa tuổi xe và log(giá) trong từng đời xe",
        subtitle=subtitle("Thanh = KTC 95% (Bonett–Wright); các nhóm có độ rộng năm khác nhau",
                          np.nansum(part["n_samples"].to_numpy(dtype=float)), version),
    )


def plot_price_by_category(
    df: pd.DataFrame, column: str, out_dir: Path, version: str | None, *,
    order: Sequence[str] | None = None, top: int = 12, question: str = "Q1",
) -> Path:
    """Boxplot giá (trục log) theo nhãn của một biến phân loại, kèm n mỗi nhãn."""
    data = df[[column, "price"]].dropna()
    labels = data[column].astype(str)
    counts = labels.value_counts()
    levels = [lvl for lvl in (order or counts.index[:top].tolist()) if lvl in counts.index]
    groups = [data.loc[labels == lvl, "price"].to_numpy(dtype=float) for lvl in levels]
    fig, (ax,) = new_figure(11, 5.2)
    ax.boxplot(groups, showfliers=False, widths=0.6, medianprops={"color": PALETTE[1], "linewidth": 2})
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(levels) + 1))
    ax.set_xticklabels([f"{lvl}\n(n={vn_number(counts[lvl])})" for lvl in levels], fontsize=BASE_FONT - 2)
    ax.set_ylabel("Giá (USD, trục log)")
    return save_figure(
        fig, out_dir, f"eda_{question}_price_by_{column}",
        title=f"{question} · Phân phối giá theo {column}",
        subtitle=subtitle("Hộp = Q1–Q3, vạch = trung vị, râu = 1,5·IQR (ẩn điểm ngoại lai)", len(data), version),
    )


SELLER_LABELS: dict[str, str] = {"dealer": "Đại lý", "private": "Chính chủ", "unknown": "Không rõ"}


def plot_condition_by_seller(table: pd.DataFrame, out_dir: Path, version: str | None, *, band: str = "6-10") -> Path:
    """Q1b · Trung vị giá theo tình trạng, tách đại lý và chính chủ, trong một nhóm tuổi."""
    part = table[table["age_band"] == band]
    sellers = [s for s in ("dealer", "private", "unknown") if s in set(part["seller_type"])]
    levels = [c for c in CONDITION_ORDER if c in set(part["condition_clean"])]
    fig, (ax,) = new_figure(11, 5.2)
    width, x = 0.8 / max(len(sellers), 1), np.arange(len(levels))
    for i, seller in enumerate(sellers):
        sub = part[part["seller_type"] == seller].set_index("condition_clean").reindex(levels)
        ax.bar(x + (i - (len(sellers) - 1) / 2) * width, sub["median_price"].to_numpy(dtype=float), width,
               color=PALETTE[i], label=SELLER_LABELS.get(seller, seller))
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Trung vị giá (USD)")
    ax.legend(frameon=False)
    return save_figure(
        fig, out_dir, "eda_Q1b_condition_by_seller",
        title=f"Q1b · Trung vị giá theo tình trạng, tách đại lý và chính chủ (xe {band} tuổi)",
        subtitle=subtitle("Phân tầng theo loại người bán và nhóm tuổi", part["n"].sum(), version),
    )


def plot_common_ranking(table: pd.DataFrame, out_dir: Path, version: str | None) -> Path:
    """Q1 · Tỷ lệ phương sai log(giá) giải thích bởi riêng từng biến (r² hoặc η²)."""
    part = table.dropna(subset=["share_explained"]).sort_values("share_explained")
    fig, (ax,) = new_figure(10, 5.6)
    colors = [PALETTE[0] if kind == "numeric" else PALETTE[1] for kind in part["kind"]]
    labels = [f"{f} ({m})" for f, m in zip(part["feature"], part["measure"])]
    ax.barh(labels, 100.0 * part["share_explained"].to_numpy(dtype=float), color=colors)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("% phương sai log(giá) giải thích bởi riêng biến")
    n = part["n_samples"].iloc[0] if len(part) else 0
    return save_figure(
        fig, out_dir, "eda_Q1_common_scale_ranking",
        title="Q1 · Liên hệ biên với log(giá): biến số (r²) và biến phân loại (η²)",
        subtitle=subtitle("Cùng một tập dòng cho mọi biến; liên hệ biên, không phải hệ số có điều kiện", n, version),
    )
