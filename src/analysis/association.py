"""M2-18 · Liên hệ giữa biến phân loại và giá, cùng thang với biến số.

- η² (tỷ số tương quan) = SS_between / SS_total = R² của hồi quy y theo các
  biến giả của nhóm. Vì vậy η² trên ``log_price`` so được trực tiếp với r²
  của biến số (cùng là "tỷ lệ phương sai giải thích bởi riêng biến đó").
- Kruskal–Wallis H trên hạng, độ lớn hiệu ứng ε² = H / (n − 1).
- Cramér's V (có hiệu chỉnh độ chệch Bergsma, 2013) cho hai biến phân loại.
- Cliff's δ = P(X > Y) − P(X < Y), tính từ thống kê U của Mann–Whitney.

Không mã hóa nhãn thành số rồi tính Pearson: thứ tự mã là tùy ý.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.correlation_engine import as_float_series, compute_series_correlation


def _paired_frame(values: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Ghép giá trị số và nhãn nhóm theo index; bỏ dòng thiếu hoặc không hữu hạn."""
    y = as_float_series(values, str(values.name or "y"))
    if y.index.equals(groups.index):
        frame = pd.DataFrame({"y": y.to_numpy(), "g": groups.to_numpy(dtype=object)})
    else:
        if not (y.index.is_unique and groups.index.is_unique):
            raise ValueError("Index trùng lặp: không ghép được giá trị với nhóm.")
        common = y.index.intersection(groups.index)
        frame = pd.DataFrame(
            {"y": y.reindex(common).to_numpy(), "g": groups.reindex(common).to_numpy(dtype=object)}
        )
    keep = np.isfinite(frame["y"].to_numpy()) & pd.notna(frame["g"]).to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def eta_squared(values: pd.Series, groups: pd.Series) -> dict[str, float]:
    """η² và ω² (ít chệch hơn khi có nhiều nhóm) của ``values`` theo ``groups``."""
    frame = _paired_frame(values, groups)
    n = len(frame)
    grouped = frame.groupby("g", sort=False)["y"]
    sizes, means = grouped.size(), grouped.mean()
    k = int(len(sizes))
    result = {"n": float(n), "k": float(k), "ss_between": math.nan, "ss_total": math.nan,
              "eta2": math.nan, "omega2": math.nan}
    if n < 2 or k < 2:
        return result
    grand = float(frame["y"].mean())
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_total = float(((frame["y"] - grand) ** 2).sum())
    result.update(ss_between=ss_between, ss_total=ss_total)
    if ss_total <= 0:
        return result
    result["eta2"] = ss_between / ss_total
    if n > k:
        ms_within = (ss_total - ss_between) / (n - k)
        result["omega2"] = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return result


def kruskal_epsilon_squared(values: pd.Series, groups: pd.Series) -> dict[str, float]:
    """Kruskal–Wallis H, p-value và ε² = H / (n − 1) (Tomczak & Tomczak, 2014)."""
    frame = _paired_frame(values, groups)
    samples = [part.to_numpy() for _, part in frame.groupby("g", sort=False)["y"]]
    n = len(frame)
    result = {"n": float(n), "k": float(len(samples)), "kw_h": math.nan, "kw_p": math.nan,
              "epsilon2": math.nan}
    if len(samples) < 2 or n < 3:
        return result
    try:
        h_stat, p_value = stats.kruskal(*samples)
    except ValueError:  # mọi giá trị bằng nhau
        return result
    result.update(kw_h=float(h_stat), kw_p=float(p_value), epsilon2=float(h_stat) / (n - 1))
    return result


def cramers_v(a: pd.Series, b: pd.Series, *, bias_correction: bool = True) -> dict[str, float]:
    """Cramér's V của hai biến phân loại (mặc định có hiệu chỉnh độ chệch)."""
    if a.index.equals(b.index):
        frame = pd.DataFrame({"a": a.to_numpy(dtype=object), "b": b.to_numpy(dtype=object)})
    else:
        common = a.index.intersection(b.index)
        frame = pd.DataFrame({"a": a.reindex(common).to_numpy(dtype=object),
                              "b": b.reindex(common).to_numpy(dtype=object)})
    frame = frame.dropna()
    table = pd.crosstab(frame["a"], frame["b"]).to_numpy(dtype=float)
    n = float(table.sum())
    rows, cols = table.shape
    result = {"n": n, "rows": float(rows), "cols": float(cols), "chi2": math.nan,
              "p_value": math.nan, "dof": math.nan, "cramers_v": math.nan}
    if rows < 2 or cols < 2 or n < 2:
        return result
    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
    result.update(chi2=float(chi2), p_value=float(p_value), dof=float(dof))
    phi2 = float(chi2) / n
    if bias_correction:
        phi2 = max(0.0, phi2 - (cols - 1) * (rows - 1) / (n - 1))
        rows_c = rows - (rows - 1) ** 2 / (n - 1)
        cols_c = cols - (cols - 1) ** 2 / (n - 1)
        denom = min(cols_c - 1, rows_c - 1)
    else:
        denom = min(cols - 1, rows - 1)
    result["cramers_v"] = math.sqrt(phi2 / denom) if denom > 0 else math.nan
    return result


def cliffs_delta(values: pd.Series, flag: pd.Series) -> dict[str, float]:
    """Cliff's δ của nhóm ``flag == True`` so với nhóm còn lại, kèm p của Mann–Whitney."""
    frame = _paired_frame(values, flag.astype("boolean"))
    group = frame["g"].astype(bool).to_numpy()
    first, second = frame.loc[group, "y"].to_numpy(), frame.loc[~group, "y"].to_numpy()
    result = {"n_true": float(first.size), "n_false": float(second.size),
              "cliffs_delta": math.nan, "mw_p": math.nan}
    if first.size == 0 or second.size == 0:
        return result
    u_first, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
    delta = 2.0 * float(u_first) / (first.size * second.size) - 1.0
    result.update(cliffs_delta=delta, mw_p=float(p_value))
    return result


def level_summary(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Trung vị và IQR theo từng nhãn; kiểm tra trung vị có tăng theo ``order``.

    Dùng làm bằng chứng cho thứ bậc mã hóa (ví dụ ``condition_clean``): một
    thứ bậc "có nghĩa" chưa chắc đơn điệu theo giá. Nhãn ngoài ``order``
    (ví dụ ``unknown``) được liệt kê sau và không tham gia kiểm tra thứ tự.
    """
    frame = _paired_frame(df[value_col], df[group_col])
    grouped = frame.groupby("g", sort=False)["y"]
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "median": grouped.median(),
            "q1": grouped.quantile(0.25),
            "q3": grouped.quantile(0.75),
        }
    )
    table.index.name = "level"
    table = table.reset_index()
    table["level"] = table["level"].astype(str)
    table["share_pct"] = 100.0 * table["n"] / table["n"].sum()
    table.insert(0, "column", group_col)
    table.insert(1, "value_col", value_col)
    if order is None:
        return table.sort_values("median").reset_index(drop=True)
    position = {str(label): i for i, label in enumerate(order)}
    table["order_position"] = table["level"].map(position)
    ordered_idx = table["order_position"].dropna().sort_values().index
    breaks = table.loc[ordered_idx, "median"].diff().lt(0)
    table["breaks_order"] = pd.Series(pd.NA, index=table.index, dtype="boolean")
    table.loc[ordered_idx, "breaks_order"] = breaks.to_numpy()
    table["order_monotonic"] = bool(not breaks.any())
    others = table.loc[table.index.difference(ordered_idx)].sort_values("n", ascending=False)
    return pd.concat([table.loc[ordered_idx], others]).reset_index(drop=True)


def categorical_association_table(
    df: pd.DataFrame,
    target: str,
    columns: Sequence[str],
    *,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """Một dòng mỗi biến phân loại: η², ω², Kruskal–Wallis ε², nhãn nhỏ nhất."""
    rows = []
    for col in columns:
        base = {"dataset_version": dataset_version, "column": col, "target": target}
        if col not in df.columns or target not in df.columns:
            rows.append({**base, "note": "Thiếu cột"})
            continue
        eta = eta_squared(df[target], df[col])
        kw = kruskal_epsilon_squared(df[target], df[col])
        counts = df[col].value_counts(dropna=True)
        rows.append({
            **base,
            "n": int(eta["n"]),
            "n_levels": int(eta["k"]),
            "smallest_level": str(counts.idxmin()) if len(counts) else "",
            "smallest_level_n": int(counts.min()) if len(counts) else 0,
            "eta2": eta["eta2"],
            "omega2": eta["omega2"],
            "kw_h": kw["kw_h"],
            "kw_p": kw["kw_p"],
            "epsilon2": kw["epsilon2"],
            "note": "",
        })
    table = pd.DataFrame(rows)
    if "eta2" not in table.columns:
        return table
    return table.sort_values("eta2", ascending=False, na_position="last").reset_index(drop=True)


def common_scale_ranking(
    df: pd.DataFrame,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """Xếp hạng chung: r² (biến số) và η² (biến phân loại) trên CÙNG tập dòng.

    Cả hai là tỷ lệ phương sai của ``target`` giải thích bởi riêng một biến
    (liên hệ biên), nên đặt cạnh nhau được. Đây không phải hệ số có điều kiện;
    xếp hạng có điều kiện dùng hàm của TV3 (``src.models.feature_ranking``).
    """
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    mask = np.isfinite(as_float_series(df[target], target).to_numpy())
    for col in numeric:
        mask &= np.isfinite(as_float_series(df[col], col).to_numpy())
    for col in categorical:
        mask &= df[col].notna().to_numpy()
    frame = df.loc[mask, [target, *numeric, *categorical]]
    n = int(len(frame))
    rows = []
    for col in numeric:
        res = compute_series_correlation(frame[col], frame[target], col, target, min_n=4)
        rows.append({"feature": col, "kind": "numeric", "measure": "r²",
                     "share_explained": res["pearson_r"] ** 2, "pearson_r": res["pearson_r"],
                     "spearman_rho": res["spearman_rho"], "n_levels": math.nan})
    for col in categorical:
        eta = eta_squared(frame[target], frame[col])
        rows.append({"feature": col, "kind": "categorical", "measure": "η²",
                     "share_explained": eta["eta2"], "pearson_r": math.nan,
                     "spearman_rho": math.nan, "n_levels": eta["k"]})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.insert(0, "dataset_version", dataset_version)
    table["target"] = target
    table["n_samples"] = n
    table["rank"] = table["share_explained"].rank(ascending=False, method="min")
    return table.sort_values("rank").reset_index(drop=True)
