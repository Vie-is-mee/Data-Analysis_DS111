"""M2-03 · Bộ máy tương quan cấp cặp ``pd.Series`` (Gate 1).

Đây là nguồn số liệu tương quan duy nhất của nhóm: mọi hệ số được tính trên
từng cặp ``Series`` bằng ``scipy.stats``, không tính ma trận tương quan của cả
bảng. Mỗi kết quả kèm số mẫu thực tế, p-value và khoảng tin cậy, vì với hàng
trăm nghìn dòng p-value gần như luôn bằng 0 và không cho biết độ mạnh.

Khoảng tin cậy (mặc định 95%):

- Pearson: biến đổi Fisher, z = artanh(r), SE = 1 / sqrt(n − 3).
- Spearman: Fisher z với SE = sqrt((1 + ρ²/2) / (n − 3)) (Bonett & Wright, 2000).
- Tương quan riêng phần: Fisher z với SE = 1 / sqrt(n − 3 − k), k là số biến kiểm soát.

Kendall τ_b chỉ dùng để kiểm tra độ bền (tùy chọn). SciPy tính τ bằng thuật
toán O(n log n) nên không chậm; Spearman được chọn làm hệ số thứ hạng chính vì
so trực tiếp được với Pearson qua chỉ số Δ = |ρ| − |r|.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_MIN_N: int = 30
DELTA_THRESHOLD: float = 0.10
SKEW_THRESHOLD: float = 1.0

RESULT_COLUMNS: tuple[str, ...] = (
    "dataset_version",
    "scope",
    "group_value",
    "x",
    "y",
    "n_samples",
    "n_dropped",
    "pearson_r",
    "pearson_p",
    "pearson_ci_low",
    "pearson_ci_high",
    "spearman_rho",
    "spearman_p",
    "spearman_ci_low",
    "spearman_ci_high",
    "kendall_tau",
    "kendall_p",
    "delta_abs",
    "skew_x",
    "skew_y",
    "recommended_method",
    "method_reason",
    "ci_level",
    "note",
)
# Các khóa do compute_series_correlation trả về (bỏ 3 cột ngữ cảnh đầu).
PAIR_KEYS: tuple[str, ...] = RESULT_COLUMNS[3:]

PARTIAL_COLUMNS: tuple[str, ...] = (
    "dataset_version",
    "scope",
    "group_value",
    "x",
    "y",
    "controls",
    "method",
    "n_samples",
    "partial_r",
    "p_value",
    "ci_low",
    "ci_high",
    "ci_level",
    "note",
)


def z_critical(ci_level: float) -> float:
    """Giá trị tới hạn chuẩn hai phía cho mức tin cậy ``ci_level``."""
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level phải nằm trong khoảng (0, 1)")
    return float(stats.norm.ppf(0.5 + ci_level / 2.0))


def fisher_ci(
    r: float,
    n: int,
    ci_level: float = 0.95,
    *,
    se: float | None = None,
    n_controls: int = 0,
) -> tuple[float, float]:
    """Khoảng tin cậy Fisher z cho một hệ số tương quan.

    Trả về (NaN, NaN) khi hệ số không hữu hạn hoặc không đủ bậc tự do
    (n − 3 − k ≤ 0). Hệ số ±1 được kẹp sát biên để artanh hữu hạn.
    """
    dof = n - 3 - n_controls
    if not np.isfinite(r) or dof <= 0:
        return (math.nan, math.nan)
    if se is None:
        se = 1.0 / math.sqrt(dof)
    clipped = float(np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15))
    z = math.atanh(clipped)
    half = z_critical(ci_level) * se
    return (math.tanh(z - half), math.tanh(z + half))


def recommend_method(
    delta_abs: float, skew_x: float, skew_y: float, *, ordinal: bool = False
) -> tuple[str, str]:
    """Quy tắc kinh nghiệm (chốt trước khi xem kết quả) chọn hệ số báo cáo chính.

    Spearman nếu có biến thứ bậc, hoặc |Δ| > 0,10, hoặc |skew| > 1 ở một biến;
    ngược lại Pearson. Δ chỉ là công cụ sàng lọc, không phải phép kiểm định.
    """
    reasons: list[str] = []
    if ordinal:
        reasons.append("có biến thứ bậc")
    if np.isfinite(delta_abs) and abs(delta_abs) > DELTA_THRESHOLD:
        reasons.append(f"|Δ| = {abs(delta_abs):.2f} > {DELTA_THRESHOLD:.2f}")
    skew = max(abs(skew_x), abs(skew_y)) if np.isfinite(skew_x) and np.isfinite(skew_y) else math.nan
    if np.isfinite(skew) and skew > SKEW_THRESHOLD:
        reasons.append(f"|skew| = {skew:.2f} > {SKEW_THRESHOLD:.1f}")
    if reasons:
        return "spearman", "; ".join(reasons)
    return "pearson", "gần tuyến tính, không lệch mạnh"


def as_float_series(series: pd.Series, name: str) -> pd.Series:
    """Chuyển Series số hoặc boolean sang float64 (NA → NaN), giữ nguyên index.

    Biến chuỗi/phân loại bị từ chối: thứ tự mã số của nhãn là tùy ý nên hệ số
    tương quan trên mã đó không có nghĩa. Biến thứ bậc phải được ánh xạ sang
    số trước, có giải thích.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} phải là pandas.Series")
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        return pd.Series(values, index=series.index, name=series.name)
    raise TypeError(
        f"{name} có kiểu {dtype}: chỉ nhận biến số hoặc boolean. "
        "Biến thứ bậc cần ánh xạ sang số trước; biến danh định dùng η² hoặc Cramér's V."
    )


def _aligned_values(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Ghép hai Series theo index, giữ cặp hữu hạn; trả về (x, y, số cặp bị loại)."""
    if x.index.equals(y.index):
        xv, yv = x.to_numpy(), y.to_numpy()
    else:
        if not (x.index.is_unique and y.index.is_unique):
            raise ValueError(
                "Index của hai Series khác nhau và có nhãn trùng: không ghép cặp an toàn được."
            )
        common = x.index.intersection(y.index)
        xv = x.reindex(common).to_numpy()
        yv = y.reindex(common).to_numpy()
    finite = np.isfinite(xv) & np.isfinite(yv)
    return xv[finite], yv[finite], int((~finite).sum())


def _empty_result(x: str, y: str, n: int, n_dropped: int, ci_level: float, note: str) -> dict:
    result: dict[str, object] = {key: math.nan for key in PAIR_KEYS}
    result.update(
        x=x,
        y=y,
        n_samples=int(n),
        n_dropped=int(n_dropped),
        recommended_method="",
        method_reason="",
        ci_level=ci_level,
        note=note,
    )
    return result


def compute_series_correlation(
    s_x: pd.Series,
    s_y: pd.Series,
    x_name: str | None = None,
    y_name: str | None = None,
    *,
    ci_level: float = 0.95,
    include_kendall: bool = False,
    min_n: int = DEFAULT_MIN_N,
    ordinal: bool = False,
) -> dict[str, object]:
    """Pearson, Spearman (+ Kendall tùy chọn) cho đúng một cặp Series.

    Các bước: ghép index (inner), bỏ cặp có NaN/±inf, đếm n; nếu n < ``min_n``
    hoặc một biến là hằng số thì trả NaN kèm ghi chú thay vì ném lỗi.
    Kết quả luôn có ``n_samples`` và khoảng tin cậy của hai hệ số chính.
    """
    x_label = str(x_name if x_name is not None else (s_x.name if s_x.name is not None else "x"))
    y_label = str(y_name if y_name is not None else (s_y.name if s_y.name is not None else "y"))
    x = as_float_series(s_x, x_label)
    y = as_float_series(s_y, y_label)
    xv, yv, n_dropped = _aligned_values(x, y)
    n = int(xv.size)
    threshold = max(int(min_n), 4)
    if n < threshold:
        return _empty_result(
            x_label, y_label, n, n_dropped, ci_level, f"n = {n} < ngưỡng {threshold}: không báo hệ số"
        )
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        return _empty_result(
            x_label, y_label, n, n_dropped, ci_level,
            "Một biến là hằng số trên các cặp hợp lệ: hệ số không xác định",
        )
    r, p_r = stats.pearsonr(xv, yv)
    rho, p_s = stats.spearmanr(xv, yv)
    r, p_r, rho, p_s = float(r), float(p_r), float(rho), float(p_s)
    result = _empty_result(x_label, y_label, n, n_dropped, ci_level, "")
    result.update(pearson_r=r, pearson_p=p_r, spearman_rho=rho, spearman_p=p_s)
    result["pearson_ci_low"], result["pearson_ci_high"] = fisher_ci(r, n, ci_level)
    se_spearman = math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))
    result["spearman_ci_low"], result["spearman_ci_high"] = fisher_ci(
        rho, n, ci_level, se=se_spearman
    )
    if include_kendall:
        tau, p_t = stats.kendalltau(xv, yv)
        result.update(kendall_tau=float(tau), kendall_p=float(p_t))
    skew_x = float(stats.skew(xv, bias=False))
    skew_y = float(stats.skew(yv, bias=False))
    delta = abs(rho) - abs(r)
    method, reason = recommend_method(delta, skew_x, skew_y, ordinal=ordinal)
    result.update(
        delta_abs=delta,
        skew_x=skew_x,
        skew_y=skew_y,
        recommended_method=method,
        method_reason=reason,
    )
    return result


def correlate_pairs(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    *,
    scope: str = "all",
    group_value: object = "all",
    dataset_version: str | None = None,
    ordinal_columns: Sequence[str] = (),
    ci_level: float = 0.95,
    include_kendall: bool = False,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Bảng dạng dài: mỗi dòng là một cặp (x, y) trong một phạm vi ``scope``."""
    ordinal_set = set(ordinal_columns)
    rows: list[dict[str, object]] = []
    for x, y in pairs:
        missing = [col for col in (x, y) if col not in df.columns]
        if missing:
            row = _empty_result(x, y, 0, 0, ci_level, f"Thiếu cột: {', '.join(missing)}")
        else:
            row = compute_series_correlation(
                df[x],
                df[y],
                x,
                y,
                ci_level=ci_level,
                include_kendall=include_kendall,
                min_n=min_n,
                ordinal=bool(ordinal_set & {x, y}),
            )
        rows.append({"dataset_version": dataset_version, "scope": scope, "group_value": str(group_value), **row})
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def _group_mask(keys: pd.Series, value: object) -> np.ndarray:
    return keys.eq(value).fillna(False).to_numpy(dtype=bool)


def correlate_by_group(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    group_col: str,
    *,
    groups: Sequence[object] | None = None,
    min_n: int = 100,
    scope: str | None = None,
    dataset_version: str | None = None,
    ordinal_columns: Sequence[str] = (),
    ci_level: float = 0.95,
    include_kendall: bool = False,
) -> pd.DataFrame:
    """Chạy ``correlate_pairs`` trong từng nhóm của ``group_col``.

    Nhóm có n < ``min_n`` vẫn có dòng kết quả (NaN + ghi chú) để người đọc
    thấy nhóm bị bỏ qua thay vì biến mất khỏi bảng.
    """
    if group_col not in df.columns:
        raise KeyError(f"Thiếu cột nhóm: {group_col}")
    pairs = list(pairs)
    keys = df[group_col]
    values = list(groups) if groups is not None else sorted(keys.dropna().unique().tolist(), key=str)
    frames = [
        correlate_pairs(
            df[_group_mask(keys, value)],
            pairs,
            scope=scope or group_col,
            group_value=value,
            dataset_version=dataset_version,
            ordinal_columns=ordinal_columns,
            ci_level=ci_level,
            include_kendall=include_kendall,
            min_n=min_n,
        )
        for value in values
    ]
    if not frames:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    controls: Sequence[str],
    *,
    method: str = "spearman",
    ci_level: float = 0.95,
    min_n: int = DEFAULT_MIN_N,
) -> dict[str, object]:
    """Tương quan riêng phần của x và y khi giữ cố định các biến ``controls``.

    Cách tính: (Spearman thì lấy hạng trước) hồi quy x và y lần lượt theo các
    biến kiểm soát (có hệ số chặn), rồi lấy tương quan của hai phần dư.
    p-value dùng phân phối t với n − 2 − k bậc tự do.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError("method phải là 'spearman' hoặc 'pearson'")
    controls = list(controls)
    columns = [x, y, *controls]
    missing = [col for col in columns if col not in df.columns]
    base: dict[str, object] = {
        "x": x,
        "y": y,
        "controls": ",".join(controls),
        "method": method,
        "n_samples": 0,
        "partial_r": math.nan,
        "p_value": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "ci_level": ci_level,
        "note": "",
    }
    if missing:
        base["note"] = f"Thiếu cột: {', '.join(missing)}"
        return base
    matrix = np.column_stack([as_float_series(df[col], col).to_numpy() for col in columns])
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    n, k = matrix.shape[0], len(controls)
    base["n_samples"] = int(n)
    if n < max(int(min_n), k + 4):
        base["note"] = f"n = {n} không đủ cho {k} biến kiểm soát"
        return base
    if method == "spearman":
        matrix = np.column_stack([stats.rankdata(matrix[:, j]) for j in range(matrix.shape[1])])
    design = np.column_stack([np.ones(n), matrix[:, 2:]])

    def residual(values: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ beta

    ex, ey = residual(matrix[:, 0]), residual(matrix[:, 1])
    denom = math.sqrt(float(ex @ ex) * float(ey @ ey))
    if denom == 0:
        base["note"] = "Phần dư hằng số: hệ số không xác định"
        return base
    r = float(np.clip(float(ex @ ey) / denom, -1.0, 1.0))
    dof = n - 2 - k
    with np.errstate(divide="ignore"):
        t_stat = r * math.sqrt(dof / max(1.0 - r * r, 1e-300))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    low, high = fisher_ci(r, n, ci_level, n_controls=k)
    base.update(partial_r=r, p_value=p_value, ci_low=low, ci_high=high)
    return base


def partial_table(
    df: pd.DataFrame,
    specs: Iterable[tuple[str, str, Sequence[str]]],
    *,
    scope: str = "all",
    group_value: object = "all",
    dataset_version: str | None = None,
    method: str = "spearman",
    ci_level: float = 0.95,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Nhiều tương quan riêng phần trong cùng một phạm vi, dạng bảng dài."""
    rows = [
        {
            "dataset_version": dataset_version,
            "scope": scope,
            "group_value": str(group_value),
            **partial_correlation(df, x, y, controls, method=method, ci_level=ci_level, min_n=min_n),
        }
        for x, y, controls in specs
    ]
    return pd.DataFrame(rows, columns=list(PARTIAL_COLUMNS))
