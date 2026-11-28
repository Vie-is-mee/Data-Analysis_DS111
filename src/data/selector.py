"""M1-10 · Bước 5 · Chọn biến → Workbook Sheet 06.

- ``to_contract``: giữ 22 cột của REQUIRED_CLEAN_SCHEMA (+ cột cờ bổ sung), ép đúng kiểu.
- ``vif_report``: VIF của các biến số (dùng hàm chung của TV3 để cùng cách tính).
- ``selection_evidence``: bảng quyết định chọn biến kèm lý do định lượng, VIF theo từng bộ biến
  và câu diễn giải. Mọi số tính trên tập **train**: căn cứ chọn biến cho mô hình không được
  dùng thông tin của val/test (cùng quy ước với TV2).

Số trong câu diễn giải sinh từ dữ liệu, không ghi cứng, nên xuất lại trên ABT V1.0 vẫn đúng.
Quyết định ở đây là ĐỀ XUẤT của TV1; "Thử" nghĩa là TV3 quyết bằng thí nghiệm CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs.project_config import REQUIRED_CLEAN_SCHEMA
from src.analysis.association import categorical_association_table, eta_squared
from src.analysis.correlation_engine import correlate_pairs
from src.analysis.eda_summary import vn_number
from src.data.transform_log import TransformLog
from src.models.feature_ranking import compute_vif_table

SHEET = "06_Selection"
# Cột bổ sung ngoài 22 cột hợp đồng (DATA_CONTRACT 1.1; validator cho phép cột thừa).
EXTRA_COLUMNS: dict[str, str] = {
    # Hợp đồng 1.1 (TV3): boolean không rỗng / string có miền cố định.
    "condition_missing": "bool",            # condition gốc rỗng, trước khi điền 'unknown'
    "cylinders_was_missing": "bool",        # xi-lanh khuyết sau tách số + xe điện = 0, trước khi điền
    "seller_type": "string",                # private / dealer / unknown
    "car_age_squared": "float64",           # M1-07: tuổi xe bậc 2 (đã lưu FormulaStore)
    # Cột truy vết/kiểm tra do TV1 thêm (TV3 đồng ý 24/11):
    "manufacturer_source": "string",        # original / title / model / model_name / none
    "manufacturer_title_conflict": "bool",  # hãng gốc khác hãng trong title
    "price_outlier_iqr": "bool",            # ngoài 1,5·IQR log(price) của hãng (ngưỡng từ train)
}
VIF_FEATURES: list[str] = ["car_age", "log_odometer", "miles_per_year", "cylinders_num", "year"]

# ── Chọn biến (M1-10) ────────────────────────────────────────────────────────
LINEAR, TREE = "OLS/Ridge", "Cây (DT/RF)"
KEEP, DROP, TRY, TARGET = "Giữ", "Bỏ", "Thử", "Mục tiêu"
# Ngưỡng kinh nghiệm (ghi ra Sheet 06 để người đọc kiểm lại được):
DUPLICATE_R = 0.99         # |r| ≥ 0,99 giữa hai biến số → trùng thông tin
MIN_SHARE = 0.01           # < 1% phương sai log_price khi đứng riêng → yếu
MIN_GAIN = 0.005           # thêm biến mà R² tăng < 0,5 điểm % → đóng góp nhỏ
MAX_ONEHOT_LEVELS = 1000   # > 1.000 nhãn → One-Hot quá thưa ở nhánh chính
CURVE_DELTA = 0.10         # |ρ| − |r| > 0,10 → quan hệ cong (cùng quy tắc với TV2)
RARE_LEVEL_ROWS = 10       # nhãn < 10 dòng train coi là hiếm
LOW_ODOMETER = 1_000       # số dặm rất nhỏ: điểm đòn bẩy của log_odometer (TV2 đã nêu)
ALL_NUMERIC = ["car_age", "car_age_squared", "year", "odometer", "log_odometer",
               "miles_per_year", "cylinders_num"]
CATEGORICAL = ["manufacturer_clean", "model_clean", "condition_clean", "title_status_clean",
               "fuel", "transmission", "drive", "type", "state", "region", "year_era",
               "seller_type", "condition_missing"]
CHECK_COLUMNS = {  # cột không phải đặc trưng → lý do (kèm η² để thấy chúng không đưa vào mô hình)
    "cylinders_was_missing": "TV3 dùng để điền lại cylinders trong từng fold, không đưa vào mô hình",
    "manufacturer_source": "Cột truy vết nguồn hãng",
    "manufacturer_title_conflict": "Cờ kiểm tra dữ liệu",
}


def to_contract(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CLEAN_SCHEMA if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột hợp đồng: {missing}")
    schema = dict(REQUIRED_CLEAN_SCHEMA) | {c: t for c, t in EXTRA_COLUMNS.items() if c in df.columns}
    out = df[list(schema)].astype(schema)
    dropped = sorted(set(df.columns) - set(out.columns))
    log.record("06_select", SHEET, f"Giữ 22 cột hợp đồng + {len(out.columns) - 22} cột cờ bổ sung",
               f"Bỏ cột trung gian/không thuộc hợp đồng: {', '.join(dropped)}", df, out)
    return out


def vif_report(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    return compute_vif_table(df, features or VIF_FEATURES)


def _v(x: float, digits: int = 2) -> str:
    return vn_number(float(x), digits)


def _pct(x: float, digits: int = 1) -> str:
    return f"{vn_number(100 * float(x), digits)}%"


def _vif_text(x: float) -> str:
    return "∞" if np.isinf(x) else _v(x, 1)


def _r2(y: np.ndarray, *blocks: pd.Series | pd.DataFrame | np.ndarray) -> float:
    """R² của OLS có hệ số chặn (bình phương tối thiểu, Bài 5) trên đúng các khối biến đã cho."""
    x = np.column_stack([np.ones(len(y))] + [np.asarray(b, dtype=float) for b in blocks])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(1 - resid.var() / y.var()) if y.var() > 0 else float("nan")


def _train(df: pd.DataFrame) -> pd.DataFrame:
    if "split_set" in df and df["split_set"].eq("train").any():
        return df[df["split_set"].eq("train")].reset_index(drop=True)
    return df.reset_index(drop=True)


def _vif_of(vif: pd.DataFrame, feature: str) -> float:
    hit = vif.loc[vif["feature"].eq(feature), "vif"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def _pearson(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def selection_evidence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Trả về (bảng quyết định, bảng VIF theo bộ biến, câu diễn giải) — tính trên tập train."""
    t = _train(df)
    y = t["log_price"].to_numpy(float)
    numeric = [c for c in ALL_NUMERIC if c in t]
    corr = correlate_pairs(t, [(c, "log_price") for c in numeric]).set_index("x")
    cats = [c for c in CATEGORICAL if c in t]
    assoc = categorical_association_table(t, "log_price", cats).set_index("column")
    rows: list[dict[str, object]] = []

    def add(col: str, group: str, measure: str, value: float | None, evidence: str,
            linear: str, tree: str) -> None:
        number = None if value is None or not np.isfinite(value) else round(float(value), 4)
        rows.append({"Cột": col, "Nhóm": group,
                     "Số giá trị khác nhau": int(t[col].nunique()) if col in t else None,
                     "Thước đo (train)": measure, "Giá trị": number,
                     "Bằng chứng định lượng": evidence, LINEAR: linear, TREE: tree})

    def r_of(col: str) -> float:
        return float(corr.loc[col, "pearson_r"]) if col in corr.index else float("nan")

    age = t["car_age"].astype(float)
    r2_age, r2_poly = _r2(y, age), _r2(y, age, age ** 2)
    r2_odo, r2_log = _r2(y, t["odometer"]), _r2(y, t["log_odometer"])
    r2_both = _r2(y, t["odometer"], t["log_odometer"])
    best_odo, other_odo = (("odometer", "log_odometer") if r2_odo >= r2_log
                           else ("log_odometer", "odometer"))
    r2_best_odo = max(r2_odo, r2_log)

    # Bộ biến số đề xuất cho OLS/Ridge (Giữ + Thử): dùng cho VIF bộ 2/3 và đo đóng góp.
    base = ["car_age", "car_age_squared", best_odo, "cylinders_num"]
    r2_base, r2_mpy = _r2(y, t[base]), _r2(y, t[base + ["miles_per_year"]])
    linear_set = base + ["miles_per_year", other_odo]
    centered = t.copy()
    centered["car_age"] = age - age.mean()
    centered["car_age_squared"] = centered["car_age"] ** 2
    vif_all = compute_vif_table(t, numeric)
    vif_lin = compute_vif_table(t, linear_set)
    vif_cen = compute_vif_table(centered, linear_set)

    # ── Biến số ──
    r, rho = r_of("car_age"), float(corr.loc["car_age", "spearman_rho"])
    delta = abs(rho) - abs(r)
    shape = "quan hệ đơn điệu nhưng cong" if delta > CURVE_DELTA else "gần tuyến tính"
    add("car_age", "Số", "r²", r ** 2,
        f"Spearman ρ = {_v(rho)}, Pearson r = {_v(r)}: |ρ| − |r| = {_v(delta)} "
        f"{'>' if delta > CURVE_DELTA else '≤'} {_v(CURVE_DELTA)} → {shape} (Bài 4)", KEEP, KEEP)

    gain = r2_poly - r2_age
    add("car_age_squared", "Số", "R² tăng thêm", gain,
        f"R² chỉ car_age = {_pct(r2_age)} → thêm car_age² = {_pct(r2_poly)} (Bài 5 · hồi quy đa thức). "
        "Cây: bình phương giữ nguyên thứ tự (car_age ≥ 0) nên cây chia y như car_age",
        KEEP if gain >= MIN_GAIN else TRY, DROP)

    r_dup = _pearson(t["year"], age)
    posting = t["posting_year"].unique() if "posting_year" in t else np.array([])
    if np.isfinite(r_dup) and abs(r_dup) >= DUPLICATE_R:
        add("year", "Số", "|r| với car_age", abs(r_dup),
            f"r(year, car_age) = {_v(r_dup, 3)}, VIF = {_vif_text(_vif_of(vif_all, 'year'))} → trùng "
            "thông tin với car_age; giữ car_age vì đọc trực tiếp là số năm khấu hao", DROP, DROP)
    else:
        add("year", "Số", "|r| với car_age", abs(r_dup),
            f"r(year, car_age) = {_v(r_dup, 3)}, chưa tới {_v(DUPLICATE_R)}", TRY, TRY)
    if len(posting) == 1:
        add("posting_year", "Số", "Số giá trị", 1,
            f"Chỉ có 1 giá trị ({int(posting[0])}) trên train → không mang thông tin; "
            "chỉ dùng để tính car_age = posting_year − year", DROP, DROP)
    elif len(posting):
        add("posting_year", "Số", "Số giá trị", len(posting),
            "Trung gian để tính car_age; thông tin đã nằm trong car_age", DROP, DROP)

    low = t["odometer"] < LOW_ODOMETER
    r_log_big = _pearson(t.loc[~low, "log_odometer"], t.loc[~low, "log_price"])
    low_price = float(t.loc[low, "price"].median()) if low.any() else float("nan")
    odo_note = {
        "odometer": f"r² riêng = {_pct(r2_odo)} (Pearson r = {_v(r_of('odometer'))})",
        "log_odometer": (f"r² riêng = {_pct(r2_log)} (Pearson r = {_v(r_of('log_odometer'))}); "
                         f"{_pct(low.mean())} dòng có odometer < {vn_number(LOW_ODOMETER)} (giá trung vị "
                         f"{vn_number(low_price)} USD) là điểm đòn bẩy của log; bỏ các dòng này thì "
                         f"r = {_v(r_log_big)}"),
    }
    add(best_odo, "Số", "r²", r2_best_odo,
        f"{odo_note[best_odo]} → mạnh hơn {other_odo} (r² = {_pct(min(r2_odo, r2_log))})", KEEP, KEEP)
    add(other_odo, "Số", "R² tăng thêm", r2_both - r2_best_odo,
        f"{odo_note[other_odo]}; thêm vào bộ đã có {best_odo}: R² {_pct(r2_best_odo, 2)} → {_pct(r2_both, 2)}. "
        f"Cây: là biến đổi đơn điệu của {best_odo} → thừa", TRY, DROP)

    mpy_gain = r2_mpy - r2_base
    add("miles_per_year", "Số", "R² tăng thêm", mpy_gain,
        f"r² riêng = {_pct(r_of('miles_per_year') ** 2)}; thêm vào ({', '.join(base)}): "
        f"R² {_pct(r2_base, 2)} → {_pct(r2_mpy, 2)}; VIF = {_vif_text(_vif_of(vif_lin, 'miles_per_year'))}",
        KEEP if mpy_gain >= MIN_GAIN else TRY, TRY)

    imputed = float(t["cylinders_was_missing"].mean()) if "cylinders_was_missing" in t else float("nan")
    add("cylinders_num", "Số", "r²", r_of("cylinders_num") ** 2,
        f"r² riêng = {_pct(r_of('cylinders_num') ** 2)}, VIF = {_vif_text(_vif_of(vif_lin, 'cylinders_num'))}; "
        f"{_pct(imputed)} giá trị đã điền → TV3 điền lại trong fold theo cylinders_was_missing", KEEP, KEEP)

    # ── Biến phân loại ── ω² = η² đã hiệu chỉnh theo số nhãn → so được cột ít nhãn với cột nhiều nhãn.
    def omega(col: str) -> float:
        return float(assoc.loc[col, "omega2"]) if col in assoc.index else float("nan")

    for col in cats:
        w, k = omega(col), int(t[col].nunique())
        smallest = int(assoc.loc[col, "smallest_level_n"]) if col in assoc.index else 0
        text = f"ω² = {_pct(w)}, {vn_number(k)} nhãn (nhãn nhỏ nhất {vn_number(smallest)} dòng)"
        group = "Cờ" if col == "condition_missing" else "Phân loại"
        if k > MAX_ONEHOT_LEVELS:
            rare = int((t[col].value_counts() < RARE_LEVEL_ROWS).sum())
            add(col, group, "ω²", w,
                f"{text}; {vn_number(rare)} nhãn có < {RARE_LEVEL_ROWS} dòng train → One-Hot quá thưa, dễ học "
                "thuộc. Mã hóa theo mục tiêu trong fold là phần mở rộng (ngoài slide)", DROP, DROP)
        elif col == "region" and "state" in t:
            one = int((t.groupby(col)["state"].nunique() == 1).sum())
            add(col, group, "ω²", w,
                f"{text}; ω² của state = {_pct(omega('state'))}; {vn_number(one)}/{vn_number(k)} vùng nằm gọn "
                "trong 1 bang → lồng trong state. Thử THAY state, không dùng cả hai", TRY, TRY)
        elif col == "year_era":
            r2_era = _r2(y, age, age ** 2, pd.get_dummies(t[col], drop_first=True))
            add(col, group, "ω²", w,
                f"{text}; thêm vào car_age + car_age²: R² {_pct(r2_poly)} → {_pct(r2_era)} (trùng một phần "
                "thông tin tuổi xe). Cây: chỉ là car_age chia khoảng → thừa", TRY, DROP)
        elif col == "seller_type":
            add(col, group, "ω²", w,
                f"{text}; cột mới của hợp đồng 1.1, TV3 chưa thử; đại lý và cá nhân bán cơ cấu xe "
                "khác nhau (xem báo cáo gộp VIN ở Sheet 03)", TRY, TRY)
        elif col == "condition_missing":
            add(col, group, "ω²", w,
                f"{text}; cờ thiếu (Bài 2 · Missingness Indicator), TV3 đồng ý thử qua CV", TRY, TRY)
        elif np.isfinite(w) and w < MIN_SHARE:
            add(col, group, "ω²", w,
                f"{text}, dưới {_pct(MIN_SHARE)} → yếu khi đứng riêng; vẫn để TV3 kiểm tra qua CV "
                "(liên hệ biên yếu chưa chắc vô ích khi có biến khác)", TRY, TRY)
        else:
            add(col, group, "ω²", w, text, KEEP, KEEP)

    # ── Cột không phải đặc trưng ──
    def eta(frame: pd.DataFrame, col: str) -> float:
        return float(eta_squared(frame["log_price"], frame[col])["eta2"])

    if "price_outlier_iqr" in t:
        e = eta(t, "price_outlier_iqr")
        add("price_outlier_iqr", "Cờ", "η²", e,
            f"Tính từ price → rò rỉ mục tiêu: η² = {_pct(e)} chính là do rò rỉ; chỉ để TV3 thử lọc train",
            DROP, DROP)
    for col, why in CHECK_COLUMNS.items():
        if col in t:
            e = eta(t, col)
            add(col, "Cờ", "η²", e, f"{why}; η² = {_pct(e, 2)}", DROP, DROP)
    if "split_set" in df and df["split_set"].nunique() > 1:
        e = eta(df, "split_set")
        add("split_set", "Chia tập", "η² (toàn ABT)", e,
            f"Cột chia tập của TV3; η² = {_v(100 * e, 4)}% ≈ 0 → chia tập không phụ thuộc giá "
            "(kiểm tra không rò rỉ)", DROP, DROP)
    add("listing_id", "Khóa", "", None, "Khóa ghép, không mang thông tin về giá", DROP, DROP)
    add("log_price", "Mục tiêu", "", None, "Biến mục tiêu khi huấn luyện, ln(price)", TARGET, TARGET)
    add("price", "Mục tiêu", "", None, "Giá USD: TV3 tính MSE/RMSE/R² trên USD sau khi đổi ngược",
        TARGET, TARGET)

    # ── VIF theo bộ biến + diễn giải ──
    vif = pd.concat([
        vif_all.assign(set="1. Ban đầu: mọi biến số"),
        vif_lin.assign(set="2. Đề xuất OLS/Ridge (Giữ + Thử)"),
        vif_cen.assign(set="3. Kiểm tra: như bộ 2 nhưng car_age trừ trung bình rồi mới bình phương"),
    ], ignore_index=True)
    notes = [
        f"Tính trên tập train, n = {vn_number(len(t))} dòng. VIF > 5 cảnh báo, > 10 nghiêm trọng (phần mở "
        "rộng ngoài slide). VIF cao làm hệ số riêng của từng biến kém ổn định nhưng không làm sai dự báo; "
        "Ridge của TV3 co hệ số nên chịu được cộng tuyến vừa phải.",
    ]
    if np.isfinite(r_dup):
        why = (f" vì mọi tin đăng năm {int(posting[0])} nên car_age = {int(posting[0])} − year"
               if len(posting) == 1 else "")
        notes.append(f"Bộ 1: year và car_age có r = {_v(r_dup, 3)}, VIF = {_vif_text(_vif_of(vif_all, 'year'))}"
                     f" / {_vif_text(_vif_of(vif_all, 'car_age'))}{why} → bỏ year.")
    notes.append(
        f"Bộ 2: car_age và car_age² có r = {_v(_pearson(age, age ** 2), 3)}, VIF = "
        f"{_vif_text(_vif_of(vif_lin, 'car_age'))} / {_vif_text(_vif_of(vif_lin, 'car_age_squared'))}. "
        f"Bộ 3 trừ trung bình car_age trước khi bình phương thì còn {_vif_text(_vif_of(vif_cen, 'car_age'))} / "
        f"{_vif_text(_vif_of(vif_cen, 'car_age_squared'))} → đây là cộng tuyến cấu trúc của đa thức, không "
        "phải hai biến trùng thông tin; vẫn giữ car_age² vì R² tăng rõ (bảng trên).")
    rest = vif_lin[~vif_lin["feature"].isin(["car_age", "car_age_squared"])]
    if len(rest) and rest["vif"].notna().any():
        top = rest.loc[rest["vif"].idxmax()]
        notes.append(f"Các biến còn lại của bộ 2 có VIF ≤ {_vif_text(float(top['vif']))} "
                     f"(cao nhất là {top['feature']}) → không cộng tuyến đáng kể.")
    vif = vif.rename(columns={"set": "Bộ biến", "feature": "Biến", "vif": "VIF", "vif_flag": "Mức",
                              "n_samples": "n"})[["Bộ biến", "Biến", "VIF", "Mức", "n"]]
    vif["VIF"] = vif["VIF"].round(3)
    return pd.DataFrame(rows), vif, notes
