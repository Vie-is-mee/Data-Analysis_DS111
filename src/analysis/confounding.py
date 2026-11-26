"""M2-19 · Biến nhiễu và cơ chế thiếu dữ liệu.

- Thiếu dữ liệu: so tỷ lệ thiếu giữa các nhóm quan sát được (bác bỏ MCAR nếu
  khác nhau rõ). Không tách được MAR với MNAR chỉ bằng dữ liệu.
- Biến nhiễu: phân tầng (ví dụ giá theo tình trạng × loại người bán × nhóm tuổi)
  và Cramér's V giữa các biến nhóm (ví dụ hãng × loại xe).
Mọi kết luận ở đây là liên hệ, không phải nhân quả.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.analysis.association import cramers_v
from src.analysis.group_analysis import FOCUS_CLUSTERS

AGE_BANDS: tuple[tuple[int, int], ...] = ((0, 5), (6, 10), (11, 15), (16, 20), (21, 100))


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Ném KeyError có danh sách cột thiếu (bộ điều phối ghi lại bước bị bỏ qua)."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột: {', '.join(missing)}")


def missingness_by(
    df: pd.DataFrame, flag_col: str, by_cols: Sequence[str], *, dataset_version: str | None = None
) -> pd.DataFrame:
    """Tỷ lệ ``flag_col`` = True theo từng nhãn của mỗi biến nhóm, kèm Cramér's V."""
    require_columns(df, [flag_col])
    flag = df[flag_col].astype("boolean").fillna(False)
    rows = []
    for col in by_cols:
        if col not in df.columns:
            continue
        labels = df[col].astype("string")
        strength = cramers_v(flag, labels)["cramers_v"]
        for level, part in flag.groupby(labels, observed=True, sort=True):
            rows.append({"dataset_version": dataset_version, "flag": flag_col, "by": col,
                         "level": str(level), "n": int(len(part)),
                         "flag_share_pct": 100.0 * float(part.astype(float).mean()),
                         "cramers_v": strength})
    return pd.DataFrame(rows)


def condition_by_seller(
    df: pd.DataFrame,
    *,
    age_bands: Sequence[tuple[int, int]] = AGE_BANDS,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """Trung vị giá theo tình trạng × loại người bán trong từng nhóm tuổi.

    Bảng này kiểm tra lời giải thích "condition để trống = người bán giấu xe":
    nếu chênh lệch biến mất khi tách theo người bán, đó là hiệu ứng cơ cấu.
    """
    require_columns(df, ["condition_clean", "seller_type", "car_age", "price"])
    rows = []
    for low, high in age_bands:
        part = df[df["car_age"].between(low, high)]
        grouped = part.groupby(["seller_type", "condition_clean"], observed=True, sort=True)["price"]
        for (seller, condition), values in grouped:
            rows.append({"dataset_version": dataset_version, "age_band": f"{low}-{high}",
                         "seller_type": str(seller), "condition_clean": str(condition),
                         "n": int(len(values)), "median_price": float(values.median())})
    return pd.DataFrame(rows)


def type_mix_by_cluster(
    df: pd.DataFrame,
    *,
    cluster_col: str = "analysis_cluster",
    clusters: Sequence[str] = FOCUS_CLUSTERS,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """Tỷ trọng loại xe trong từng cụm hãng (cụm khác cơ cấu → so sánh bị nhiễu)."""
    require_columns(df, [cluster_col, "type"])
    part = df[df[cluster_col].isin(list(clusters)).to_numpy(dtype=bool)]
    counts = pd.crosstab(part[cluster_col].astype(str).rename("cluster"),
                         part["type"].astype(str).rename("type"))
    shares = counts.div(counts.sum(axis=1), axis=0) * 100.0
    long = shares.reset_index().melt(id_vars="cluster", var_name="type", value_name="share_pct")
    long["n"] = counts.reset_index().melt(id_vars="cluster", var_name="type", value_name="n")["n"].to_numpy()
    long.insert(0, "dataset_version", dataset_version)
    return long


def association_pairs(
    df: pd.DataFrame, pairs: Sequence[tuple[str, str]], *, dataset_version: str | None = None
) -> pd.DataFrame:
    """Cramér's V cho các cặp biến phân loại đã chọn trước."""
    rows = []
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            rows.append({"dataset_version": dataset_version, "a": a, "b": b, "note": "Thiếu cột"})
            continue
        rows.append({"dataset_version": dataset_version, "a": a, "b": b,
                     **cramers_v(df[a].astype("string"), df[b].astype("string")), "note": ""})
    return pd.DataFrame(rows)


def cylinders_missing_vs_observed(df: pd.DataFrame, *, dataset_version: str | None = None) -> pd.DataFrame:
    """So giá khi số xi-lanh được khai báo và khi bị thiếu (thiếu dữ liệu có mang thông tin không)."""
    require_columns(df, ["cylinders_was_missing", "price"])
    flag = df["cylinders_was_missing"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    scopes = [("all", "all", slice(None))]
    if "seller_type" in df.columns:
        scopes += [("seller_type", str(v), df["seller_type"].eq(v).fillna(False).to_numpy(dtype=bool))
                   for v in sorted(df["seller_type"].dropna().unique().tolist(), key=str)]
    rows = []
    for scope, value, mask in scopes:
        part_flag = flag[mask]
        prices = df["price"].to_numpy(dtype=float)[mask]
        rows.append({
            "dataset_version": dataset_version, "scope": scope, "group_value": value,
            "n": int(part_flag.size),
            "missing_share_pct": 100.0 * float(part_flag.mean()) if part_flag.size else float("nan"),
            "median_price_observed": float(pd.Series(prices[~part_flag]).median()),
            "median_price_missing": float(pd.Series(prices[part_flag]).median()),
        })
    return pd.DataFrame(rows)
