"""Đặc trưng miền xe cũ & Mã hóa Phân cấp Phần dư Hao mòn (Exp5 TV3).

Ý tưởng cốt lõi (khớp tư duy tách tác động của Bài 04 & Hồi quy Bài 05):
1. Khấu hao gãy khúc 3 vùng tuổi (`age_hinge_15`, `age_hinge_25`):
   nắm đúng đáy giá 20–25 tuổi và nhánh tăng giá xe cổ >25 tuổi mà vẫn tuyến tính theo hệ số.
2. Chuẩn hóa gốc dòng xe (`model_stem`) & tín hiệu phiên bản/tải nặng từ `model_clean`:
   gom ~20.000 chuỗi dòng xe phân mảnh về ~3.600 gốc chuẩn + 6 cờ phân khúc thực tế.
3. Mã hóa Phân cấp Phần dư Hao mòn theo nhóm xe (`WearAdjustedHierarchicalEncoder`):
   - Bước 1: trong từng fold `train`, khớp đường hao mòn cơ sở từ tuổi/số dặm/xi-lanh
     rồi lấy phần dư `r = log_price - y_wear_hat` (giá trị nội tại sau khi đã trừ hao mòn).
   - Bước 2: co mượt Bayes phân cấp 4 tầng (Toàn tập -> Hãng -> Hãng|Kiểu -> Hãng|Gốc -> Hãng|Dòng)
     và địa lý (Toàn tập -> Bang -> Vùng) hoàn toàn Out-of-Fold theo `GroupKFold(vehicle_group_id)`.
   - Bước 3: bổ sung độ lệch số dặm so với trung vị xe cùng tuổi (`rel_log_odometer_by_age`),
     tần suất & độ phân tán giá nội tại của gốc dòng xe, và tương tác tuổi × mỏ neo giá trị.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from configs.project_config import CV_FOLDS, RANDOM_STATE

DOMAIN_HINGE_FEATURES: tuple[str, ...] = ("age_hinge_15", "age_hinge_25")
DOMAIN_TOKEN_FEATURES: tuple[str, ...] = (
    "tok_heavy_duty",
    "tok_performance",
    "tok_luxury_trim",
    "tok_commercial_van",
    "tok_awd_4wd",
    "tok_hybrid_ev",
)
FLAT_OOF_FEATURES: tuple[str, ...] = ("flat_model_te", "flat_region_te")
WEAR_ADJ_HIER_FEATURES: tuple[str, ...] = (
    "wear_adj_stem_premium",
    "wear_adj_model_premium",
    "wear_adj_region_premium",
    "rel_log_odometer_by_age",
    "stem_log_freq",
    "stem_price_disp",
    "age_x_model_anchor",
    "age_x_log_odometer",
)

_WEAR_BASELINE_COLS: tuple[str, ...] = (
    "car_age",
    "car_age_squared",
    "age_hinge_15",
    "age_hinge_25",
    "log_odometer",
    "cylinders_num",
)

_STEM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bf[\s\-]?150\b"), "f150"),
    (re.compile(r"\bf[\s\-]?250\b"), "f250"),
    (re.compile(r"\bf[\s\-]?350\b"), "f350"),
    (re.compile(r"\bf[\s\-]?450\b"), "f450"),
    (re.compile(r"\bf[\s\-]?550\b"), "f550"),
    (re.compile(r"\bcr[\s\-]?v\b"), "crv"),
    (re.compile(r"\bhr[\s\-]?v\b"), "hrv"),
    (re.compile(r"\brav[\s\-]?4\b"), "rav4"),
    (re.compile(r"\bcx[\s\-]?([0-9]+)\b"), r"cx\1"),
    (re.compile(r"\bgrand\s+cherokee\b"), "grand_cherokee"),
    (re.compile(r"\bland\s+cruiser\b"), "land_cruiser"),
    (re.compile(r"\brange\s+rover\b"), "range_rover"),
    (re.compile(r"\bmodel\s+([3sxy])\b"), r"model_\1"),
    (re.compile(r"\b(\d)\s+series\b"), r"\1_series"),
    (re.compile(r"\b([a-z])\s*-\s*class\b"), r"\1_class"),
    (re.compile(r"\bsanta\s+fe\b"), "santa_fe"),
    (re.compile(r"\btown\s+(?:and|&)\s+country\b"), "town_country"),
    (re.compile(r"\bfj\s+cruiser\b"), "fj_cruiser"),
    (re.compile(r"\bpt\s+cruiser\b"), "pt_cruiser"),
)
_NON_ALNUM = re.compile(r"[^a-z0-9_]+")

_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "tok_heavy_duty": re.compile(
        r"\b(?:2500|3500|4500|5500|f250|f-250|f350|f-350|f450|f-450|f550|super\s*duty|hd|dually|cummins|powerstroke|duramax)\b"
    ),
    "tok_performance": re.compile(
        r"\b(?:amg|gt|gtr|shelby|corvette|z06|hellcat|srt|scat|m3|m4|m5|rs|sti|wrx|type\s*r|911|cayman|boxster|raptor|trx|rubicon|trd\s*pro)\b"
    ),
    "tok_luxury_trim": re.compile(
        r"\b(?:limited|platinum|denali|king\s*ranch|lariat|high\s*country|overland|summit|prestige|touring|reserve|premier)\b"
    ),
    "tok_commercial_van": re.compile(
        r"\b(?:sprinter|transit|promaster|savana|express|nv200|nv2500|nv3500)\b"
    ),
    "tok_awd_4wd": re.compile(r"\b(?:4x4|4wd|awd|quattro|xdrive|4matic)\b"),
    "tok_hybrid_ev": re.compile(r"\b(?:hybrid|plug|phev|electric|prime|e-tron)\b"),
}


def extract_model_stem(series: pd.Series) -> pd.Series:
    """Rút gọn chuỗi model_clean phân mảnh về gốc dòng xe chuẩn."""

    def _normalize_one(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        text = str(value).lower().strip()
        if not text or text == "unknown":
            return "unknown"
        for pattern, replacement in _STEM_PATTERNS:
            text = pattern.sub(replacement, text)
        text = _NON_ALNUM.sub(" ", text).strip()
        tokens = text.split()
        return tokens[0] if tokens else "unknown"

    return series.map(_normalize_one).astype("string")


def add_domain_depreciation_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng xác định (không dùng nhãn giá): khớp gãy tuổi, gốc dòng xe, cờ phiên bản."""
    out = frame.copy()
    age = out["car_age"].astype(float)
    if "car_age_squared" not in out:
        out["car_age_squared"] = age.pow(2)
    out["age_hinge_15"] = np.maximum(age - 15.0, 0.0)
    out["age_hinge_25"] = np.maximum(age - 25.0, 0.0)
    models = (
        out["model_clean"]
        if "model_clean" in out
        else pd.Series("unknown", index=out.index, dtype="string")
    )
    out["model_stem"] = extract_model_stem(models)
    lowered = models.astype("string").fillna("unknown").str.lower()
    for column, pattern in _TOKEN_PATTERNS.items():
        out[column] = lowered.str.contains(pattern, regex=True, na=False).astype(
            np.float32
        )
    return out


def _smooth_toward_parent(
    keys: pd.Series, target: np.ndarray, parent: np.ndarray, smoothing: float
) -> dict[str, float]:
    table = pd.DataFrame({"k": keys.to_numpy(), "t": target, "p": parent})
    agg = table.groupby("k", sort=False).agg(
        cnt=("t", "size"), total=("t", "sum"), prior=("p", "mean")
    )
    values = (agg["total"] + smoothing * agg["prior"]) / (agg["cnt"] + smoothing)
    return values.to_dict()


class WearAdjustedHierarchicalEncoder(TransformerMixin, BaseEstimator):
    """Mã hóa Out-of-Fold theo nhóm xe cho `model_clean` và `region`.

    Hỗ trợ hai chế độ để đối chứng công bằng trên cùng fold:
    - ``flat_oof``: Target Encoding phẳng chuẩn hướng về trung bình toàn tập.
    - ``wear_adjusted_hierarchical``: Tách phần dư hao mòn vật lý rồi co mượt
      phân cấp 4 tầng (Hãng -> Hãng|Kiểu -> Hãng|Gốc -> Hãng|Dòng) và Bang -> Vùng.
    """

    required_columns: tuple[str, ...] = (
        "manufacturer_clean",
        "type",
        "model_clean",
        "model_stem",
        "state",
        "region",
        "car_age",
        "car_age_squared",
        "age_hinge_15",
        "age_hinge_25",
        "log_odometer",
        "cylinders_num",
    )

    def __init__(
        self,
        mode: str = "wear_adjusted_hierarchical",
        *,
        m_coarse: float = 10.0,
        m_fine: float = 15.0,
        wear_alpha: float = 10.0,
        n_splits: int = CV_FOLDS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.mode = mode
        self.m_coarse = m_coarse
        self.m_fine = m_fine
        self.wear_alpha = wear_alpha
        self.n_splits = n_splits
        self.random_state = random_state

    def _validate_mode(self) -> None:
        if self.mode not in ("flat_oof", "wear_adjusted_hierarchical"):
            raise ValueError(f"Chế độ mã hóa không hợp lệ: {self.mode!r}")

    @staticmethod
    def _ensure_keys(frame: pd.DataFrame) -> tuple[pd.Series, ...]:
        make = frame["manufacturer_clean"].astype("string").fillna("unknown").astype(str)
        body = frame["type"].astype("string").fillna("unknown").astype(str)
        model = frame["model_clean"].astype("string").fillna("unknown").astype(str)
        stem = (
            frame["model_stem"].astype("string").fillna("unknown").astype(str)
            if "model_stem" in frame
            else extract_model_stem(frame["model_clean"]).astype(str)
        )
        state = frame["state"].astype("string").fillna("unknown").astype(str)
        region = frame["region"].astype("string").fillna("unknown").astype(str)
        return (
            make,
            make + "|" + body,
            make + "|" + stem,
            make + "|" + model,
            state,
            state + "|" + region,
        )

    def _fit_state(self, frame: pd.DataFrame, target: np.ndarray) -> dict:
        make, make_type, make_stem, make_model, state, region = self._ensure_keys(frame)
        if self.mode == "flat_oof":
            global_mean = float(target.mean())
            prior = np.full(len(target), global_mean)
            return {
                "global_mean": global_mean,
                "model_map": _smooth_toward_parent(make_model, target, prior, self.m_fine),
                "region_map": _smooth_toward_parent(region, target, prior, self.m_fine),
            }

        wear_cols = [c for c in _WEAR_BASELINE_COLS if c in frame]
        wear_pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=self.wear_alpha),
        )
        wear_pipe.fit(frame[wear_cols], target)
        residual = target - wear_pipe.predict(frame[wear_cols])

        global_mean = float(residual.mean())
        p0 = np.full(len(residual), global_mean)
        map_make = _smooth_toward_parent(make, residual, p0, self.m_coarse)
        p1 = make.map(map_make).fillna(global_mean).to_numpy()
        map_mt = _smooth_toward_parent(make_type, residual, p1, self.m_coarse)
        p2 = make_type.map(map_mt).fillna(pd.Series(p1, index=frame.index)).to_numpy()
        map_ms = _smooth_toward_parent(make_stem, residual, p2, self.m_fine)
        p3 = make_stem.map(map_ms).fillna(pd.Series(p2, index=frame.index)).to_numpy()
        map_mm = _smooth_toward_parent(make_model, residual, p3, self.m_fine)

        map_state = _smooth_toward_parent(state, residual, p0, self.m_coarse)
        ps = state.map(map_state).fillna(global_mean).to_numpy()
        map_reg = _smooth_toward_parent(region, residual, ps, self.m_fine)

        age_med_odo = frame.groupby("car_age")["log_odometer"].median().to_dict()
        global_med_odo = float(frame["log_odometer"].median())
        stem_counts = make_stem.value_counts().to_dict()

        global_std = float(np.std(residual)) if len(residual) > 1 else 0.0
        stem_stats = (
            pd.DataFrame({"k": make_stem.to_numpy(), "r": residual})
            .groupby("k")["r"]
            .agg(["size", "std"])
        )
        stem_disp = {
            key: float(
                (
                    row["size"] * (row["std"] if np.isfinite(row["std"]) else global_std)
                    + self.m_fine * global_std
                )
                / (row["size"] + self.m_fine)
            )
            for key, row in stem_stats.iterrows()
        }
        return {
            "global_mean": global_mean,
            "make": map_make,
            "mt": map_mt,
            "ms": map_ms,
            "mm": map_mm,
            "state": map_state,
            "reg": map_reg,
            "age_med_odo": age_med_odo,
            "global_med_odo": global_med_odo,
            "stem_counts": stem_counts,
            "stem_disp": stem_disp,
            "global_std": global_std,
        }

    def _transform_with_state(self, frame: pd.DataFrame, state_dict: dict) -> pd.DataFrame:
        make, make_type, make_stem, make_model, state, region = self._ensure_keys(frame)
        g = state_dict["global_mean"]
        if self.mode == "flat_oof":
            encoded = pd.DataFrame(
                {
                    "flat_model_te": make_model.map(state_dict["model_map"])
                    .fillna(g)
                    .to_numpy(dtype=float),
                    "flat_region_te": region.map(state_dict["region_map"])
                    .fillna(g)
                    .to_numpy(dtype=float),
                },
                index=frame.index,
            )
            return pd.concat([frame, encoded], axis=1)

        v_make = make.map(state_dict["make"]).fillna(g)
        v_mt = make_type.map(state_dict["mt"]).fillna(v_make)
        v_ms = make_stem.map(state_dict["ms"]).fillna(v_mt)
        v_mm = make_model.map(state_dict["mm"]).fillna(v_ms)
        v_state = state.map(state_dict["state"]).fillna(g)
        v_reg = region.map(state_dict["reg"]).fillna(v_state)

        expected_odo = (
            frame["car_age"]
            .map(state_dict["age_med_odo"])
            .fillna(state_dict["global_med_odo"])
            .to_numpy(dtype=float)
        )
        log_odo = frame["log_odometer"].to_numpy(dtype=float)
        rel_odo = log_odo - expected_odo
        age = frame["car_age"].to_numpy(dtype=float)
        stem_freq = np.log1p(
            make_stem.map(state_dict["stem_counts"]).fillna(0.0).to_numpy(dtype=float)
        )
        stem_disp = (
            make_stem.map(state_dict["stem_disp"])
            .fillna(state_dict["global_std"])
            .to_numpy(dtype=float)
        )

        encoded = pd.DataFrame(
            {
                "wear_adj_stem_premium": v_ms.to_numpy(dtype=float),
                "wear_adj_model_premium": v_mm.to_numpy(dtype=float),
                "wear_adj_region_premium": v_reg.to_numpy(dtype=float),
                "rel_log_odometer_by_age": rel_odo,
                "stem_log_freq": stem_freq,
                "stem_price_disp": stem_disp,
                "age_x_model_anchor": age * (v_ms.to_numpy(dtype=float) - g),
                "age_x_log_odometer": age * log_odo,
            },
            index=frame.index,
        )
        return pd.concat([frame, encoded], axis=1)

    def _validate_target(self, X: pd.DataFrame, y) -> np.ndarray:
        target = np.asarray(y, dtype=float)
        if target.shape != (len(X),) or not len(X) or not np.isfinite(target).all():
            raise ValueError("Cần nhãn mục tiêu hữu hạn khớp số dòng huấn luyện.")
        return target

    def fit(self, X: pd.DataFrame, y, groups=None):
        self._validate_mode()
        target = self._validate_target(X, y)
        self.state_ = self._fit_state(X, target)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, groups=None, **fit_params):
        self._validate_mode()
        target = self._validate_target(X, y)
        if groups is not None:
            group_series = pd.Series(groups).reset_index(drop=True)
            n_unique = int(group_series.nunique())
            splits = min(self.n_splits, n_unique)
            if splits >= 2:
                splitter = GroupKFold(n_splits=splits).split(X, target, groups=group_series)
            else:
                splitter = KFold(
                    min(self.n_splits, len(X)),
                    shuffle=True,
                    random_state=self.random_state,
                ).split(X)
        else:
            splitter = KFold(
                min(self.n_splits, len(X)),
                shuffle=True,
                random_state=self.random_state,
            ).split(X)

        parts = []
        order = []
        for tr_idx, va_idx in splitter:
            fold_state = self._fit_state(X.iloc[tr_idx], target[tr_idx])
            parts.append(self._transform_with_state(X.iloc[va_idx], fold_state))
            order.append(va_idx)
        self.state_ = self._fit_state(X, target)
        stacked = pd.concat(parts, axis=0)
        return stacked.iloc[np.argsort(np.concatenate(order))]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "state_")
        return self._transform_with_state(X, self.state_)
