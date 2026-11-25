"""M2 · Chạy toàn bộ phân tích của TV2 trên một ABT bằng một lệnh.

    python -X utf8 -m src.analysis.run_m2 --abt data/processed/used_cars_clean_v1.parquet ^
        --dataset-version v1.0 --transform-log data/processed/m1_transform_log.csv --bi

Đầu ra mặc định ở ``outputs/reports/m2/`` (Git bỏ qua): bảng CSV, hình 300 DPI,
``m2_key_numbers.csv``, ``run_manifest.json`` và bảng Power BI trong ``bi/`` (cờ ``--bi``).
Chỉ đọc ABT của TV1 và nhật ký biến đổi; không đọc lại CSV gốc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from configs.project_config import CONDITION_ORDINAL, CONFIG, TITLE_STATUS_ORDINAL_EXPERIMENTAL
from src.analysis import association, bi_export, confounding, eda_summary, figures
from src.analysis import group_analysis as ga
from src.analysis.correlation_engine import correlate_by_group, correlate_pairs, partial_table
from src.analysis.key_numbers import KeyNumbers

DEFAULT_OUT = CONFIG.reports_dir / "m2"
REQUIRED_COLUMNS: tuple[str, ...] = (
    "listing_id", "price", "log_price", "year", "car_age", "odometer", "log_odometer", "miles_per_year",
    "manufacturer_clean", "condition_clean", "title_status_clean", "cylinders_num", "fuel",
    "transmission", "drive", "type", "state", "year_era",
)
NUMERIC_X: tuple[str, ...] = ("car_age", "odometer", "log_odometer", "miles_per_year", "cylinders_observed")
RANKING_NUMERIC: tuple[str, ...] = ("car_age", "odometer", "log_odometer", "miles_per_year")
CATEGORICAL: tuple[str, ...] = (
    "manufacturer_clean", "type", "condition_clean", "title_status_clean", "drive", "fuel",
    "transmission", "state", "year_era", "seller_type",
)
MAIN_PAIRS: tuple[tuple[str, str], ...] = (("car_age", "log_price"), ("odometer", "log_price"))
ERA_ORDER: tuple[str, ...] = ("Classic", "Old", "Mid", "Modern")


class RunLog:
    """Các bước đã chạy/bỏ qua và danh sách đầu ra, ghi vào ``run_manifest.json``."""

    def __init__(self) -> None:
        self.steps: list[dict[str, str]] = []
        self.outputs: list[str] = []

    def step(self, name: str, status: str, detail: str = "") -> None:
        self.steps.append({"step": name, "status": status, "detail": detail})
        print(f"[{status}] {name}" + (f" · {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(item["status"] == "FAIL" for item in self.steps)


def write_table(table: pd.DataFrame, out_dir: Path, name: str, log: RunLog) -> None:
    path = out_dir / f"{name}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    log.outputs.append(path.name)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_abt(path: Path, log: RunLog) -> pd.DataFrame:
    """Đọc ABT, kiểm tra cột cần thiết, bỏ cột nhạy cảm, chuẩn bị cờ và xi-lanh quan sát."""
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"ABT thiếu cột M2 cần: {missing}")
    if not df["listing_id"].is_unique:
        raise ValueError("ABT có listing_id trùng")
    leaked = sorted(bi_export.SENSITIVE_COLUMNS & set(df.columns))
    if leaked:
        df = df.drop(columns=leaked)
        log.step("Bỏ cột nhạy cảm", "WARN", ", ".join(leaked))
    if "condition_missing" not in df.columns:
        df["condition_missing"] = df["condition_clean"].astype("string").eq("unknown").fillna(False).astype(bool)
        log.step("condition_missing", "INFO", "ABT thiếu cờ: suy ra từ condition_clean == 'unknown'")
    observed = df["cylinders_num"].astype("float64")
    if "cylinders_was_missing" in df.columns:
        observed = observed.mask(df["cylinders_was_missing"].astype("boolean").fillna(False).to_numpy(dtype=bool))
    else:
        log.step("cylinders_observed", "WARN", "ABT thiếu cylinders_was_missing: gồm cả giá trị đã điền")
    df["cylinders_observed"] = observed
    return df.reset_index(drop=True)


def run_correlations(df: pd.DataFrame, version: str, *, kendall: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bảng tương quan dạng dài cho mọi phạm vi và bảng tương quan riêng phần."""
    pairs = [(x, y) for x in NUMERIC_X for y in ("price", "log_price")] + [("car_age", "odometer")]
    window = df[df["car_age"].between(*ga.AGE_WINDOW)]
    frames = [
        correlate_pairs(df, pairs, scope="all", dataset_version=version, include_kendall=kendall),
        correlate_pairs(df[df["odometer"] >= 1000], pairs, scope="odometer>=1000", dataset_version=version),
        correlate_by_group(df, MAIN_PAIRS, "year_era", groups=ERA_ORDER, dataset_version=version),
        correlate_by_group(window, MAIN_PAIRS, "manufacturer_clean", groups=ga.FOCUS_BRANDS,
                           scope="brand|age1-20", dataset_version=version),
        correlate_by_group(window, MAIN_PAIRS, "analysis_cluster", groups=ga.FOCUS_CLUSTERS,
                           scope="cluster|age1-20", dataset_version=version),
    ]
    if "seller_type" in df.columns:
        frames.append(correlate_by_group(df, MAIN_PAIRS, "seller_type", dataset_version=version))
    specs = [("odometer", "log_price", ["car_age"]), ("car_age", "log_price", ["odometer"]),
             ("miles_per_year", "log_price", ["car_age"])]
    return pd.concat(frames, ignore_index=True), partial_table(df, specs, dataset_version=version)


def run_groups(df: pd.DataFrame, version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Độ dốc khấu hao (toàn bộ, cụm, hãng, trong cùng loại xe), đường trung vị và miền đời xe."""
    slopes = pd.concat([
        ga.depreciation_table(df, None, dataset_version=version),
        ga.depreciation_table(df, "analysis_cluster", ga.FOCUS_CLUSTERS, group_type="cluster",
                              dataset_version=version),
        ga.depreciation_table(df, "manufacturer_clean", ga.FOCUS_BRANDS, group_type="brand",
                              dataset_version=version),
        ga.within_type_slopes(df, dataset_version=version),
    ], ignore_index=True)
    focus = df[df["analysis_cluster"].isin(list(ga.FOCUS_CLUSTERS)).to_numpy(dtype=bool)]
    curves = pd.concat([
        ga.median_curve(df, x="car_age", y="price"),
        ga.median_curve(focus, x="car_age", y="price", by="analysis_cluster", x_range=ga.AGE_WINDOW),
    ], ignore_index=True)
    return slopes, curves, ga.era_ranges(df)


def run_association(df: pd.DataFrame, version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """η²/ε² của biến phân loại, trung vị theo nhãn (kiểm tra thứ bậc), xếp hạng cùng thang."""
    cats = [col for col in CATEGORICAL if col in df.columns]
    table = association.categorical_association_table(df, "log_price", cats, dataset_version=version)
    parts = [
        association.level_summary(df, "price", "condition_clean", order=list(CONDITION_ORDINAL)),
        association.level_summary(df, "price", "title_status_clean",
                                  order=list(TITLE_STATUS_ORDINAL_EXPERIMENTAL)),
        association.level_summary(df, "price", "type"),
    ]
    if "seller_type" in df.columns:
        parts.append(association.level_summary(df, "price", "seller_type"))
    levels = pd.concat(parts, ignore_index=True)
    levels.insert(0, "dataset_version", version)
    ranking = association.common_scale_ranking(df, "log_price", RANKING_NUMERIC, cats, dataset_version=version)
    return table, levels, ranking


def run_confounding(df: pd.DataFrame, version: str, log: RunLog) -> dict[str, pd.DataFrame]:
    """Các bảng biến nhiễu; bảng nào thiếu cột đầu vào thì ghi SKIP thay vì dừng."""
    pairs = [("manufacturer_clean", "type"), ("analysis_cluster", "type"), ("condition_missing", "seller_type"),
             ("seller_type", "year_era"), ("drive", "type")]
    jobs = {
        "confounding_condition_missing": lambda: confounding.missingness_by(
            df, "condition_missing", ["seller_type", "year_era", "analysis_cluster"], dataset_version=version),
        "confounding_cylinders_missing": lambda: confounding.missingness_by(
            df, "cylinders_was_missing", ["seller_type", "year_era", "fuel"], dataset_version=version),
        "confounding_condition_by_seller": lambda: confounding.condition_by_seller(df, dataset_version=version),
        "confounding_type_mix": lambda: confounding.type_mix_by_cluster(df, dataset_version=version),
        "confounding_cramers_v": lambda: confounding.association_pairs(df, pairs, dataset_version=version),
        "confounding_cylinders_price": lambda: confounding.cylinders_missing_vs_observed(
            df, dataset_version=version),
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, job in jobs.items():
        try:
            tables[name] = job()
            log.step(name, "OK", f"{len(tables[name])} dòng")
        except KeyError as exc:
            log.step(name, "SKIP", str(exc))
    return tables


def run_ranking(df: pd.DataFrame, version: str, log: RunLog) -> pd.DataFrame | None:
    """Ranking có điều kiện của TV3 trên tập train, kèm liên hệ biên của TV2."""
    if "split_set" not in df.columns or not df["split_set"].eq("train").any():
        log.step("feature_ranking_m2", "SKIP", "ABT không có dòng split_set = train")
        return None
    from src.analysis.ranking import ranking_with_marginal

    train = df[df["split_set"].eq("train").fillna(False).to_numpy(dtype=bool)]
    table = ranking_with_marginal(train, dataset_version=version)
    log.step("feature_ranking_m2", "OK", f"train n = {int(table['n_samples'].iloc[0]):,}")
    return table


def slug(text: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(text).lower()).strip("_")


def build_key_numbers(version: str, df: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Số liệu chính có khóa cố định: nguồn của mọi con số đưa lên slide, báo cáo, dashboard."""
    kn = KeyNumbers(version)
    kn.add("n_listings", len(df), "tin", "abt")
    for stat, unit in (("median", "USD"), ("q1", "USD"), ("q3", "USD"), ("skew", ""), ("skew_log", "")):
        kn.add_from(t["eda_numeric"], {"column": "price"}, stat, f"price_{stat}", unit, "eda_numeric")
    kn.add_from(t["eda_numeric"], {"column": "car_age"}, "median", "car_age_median", "năm", "eda_numeric")
    if "seller_type" in df.columns:
        kn.add("share_dealer_pct", 100.0 * df["seller_type"].eq("dealer").mean(), "%", "abt")
    if "confounding_condition_missing" in t:
        for seller in ("dealer", "private"):
            kn.add_from(t["confounding_condition_missing"], {"by": "seller_type", "level": seller},
                        "flag_share_pct", f"condition_missing_pct_{seller}", "%", "confounding_condition_missing")
    corr = t["correlation_results"]
    for x, y in (("car_age", "log_price"), ("odometer", "log_price"), ("log_odometer", "log_price"),
                 ("car_age", "price"), ("odometer", "price"), ("miles_per_year", "log_price"),
                 ("cylinders_observed", "log_price"), ("car_age", "odometer")):
        for col, prefix in (("pearson_r", "r"), ("spearman_rho", "rho"), ("n_samples", "n")):
            kn.add_from(corr, {"scope": "all", "x": x, "y": y}, col, f"{prefix}_{x}_{y}", "", "correlation_results")
    kn.add_from(corr, {"scope": "all", "x": "car_age", "y": "log_price"}, "kendall_tau",
                "kendall_tau_car_age_log_price", "", "correlation_results")
    for era in ERA_ORDER:
        kn.add_from(corr, {"scope": "year_era", "group_value": era, "x": "car_age", "y": "log_price"},
                    "spearman_rho", f"rho_car_age_log_price_era_{slug(era)}", "", "correlation_results")
    for _, row in t["partial_correlations"].iterrows():
        kn.add(f"partial_rho_{row['x']}_{row['y']}_given_{slug(row['controls'])}", row["partial_r"], "",
               "partial_correlations")
    for _, row in t["group_depreciation"].iterrows():
        name = f"{slug(row['group_type'])}_{slug(row['group_value'])}_{row['controls']}"
        kn.add(f"pct_per_year_{name}", row["pct_per_year"], "%/năm", "group_depreciation")
        if row["controls"] != "none":
            kn.add(f"pct_per_10k_miles_{name}", row["pct_per_10k_miles"], "%/10.000 dặm", "group_depreciation")
    for _, row in t["categorical_association"].iterrows():
        kn.add(f"eta2_{row['column']}_log_price", row.get("eta2"), "", "categorical_association")
    levels = t["level_summaries"]
    for _, row in levels[levels["column"] == "condition_clean"].iterrows():
        kn.add(f"median_price_condition_{slug(row['level'])}", row["median"], "USD", "level_summaries")
    return kn.to_frame()


def make_figures(out_dir: Path, version: str, df: pd.DataFrame, t: dict[str, pd.DataFrame], log: RunLog) -> None:
    """Vẽ toàn bộ hình; một hình lỗi được ghi FAIL nhưng không chặn các hình còn lại."""
    fig_dir = out_dir / "figures"
    corr, slopes, curves = t["correlation_results"], t["group_depreciation"], t["median_curves"]
    title_order = [*TITLE_STATUS_ORDINAL_EXPERIMENTAL, "unknown"]
    jobs = {
        "price_distribution": lambda: figures.plot_price_distribution(df, fig_dir, version),
        "age_curve": lambda: figures.plot_age_curve(curves[curves["group_col"] == "all"], fig_dir, version),
        "pearson_vs_spearman": lambda: figures.plot_correlation_comparison(
            corr[corr["scope"] == "all"], NUMERIC_X, fig_dir, version),
        "cluster_curves": lambda: figures.plot_cluster_curves(
            curves[curves["group_col"] == "analysis_cluster"], fig_dir, version),
        "brand_forest": lambda: figures.plot_depreciation_forest(
            slopes[slopes["group_type"] == "brand"], ga.FOCUS_BRANDS, fig_dir, version),
        "cluster_forest": lambda: figures.plot_depreciation_forest(
            slopes[slopes["group_type"] == "cluster"], ga.FOCUS_CLUSTERS, fig_dir, version,
            name="eda_Q2_depreciation_by_cluster", title="Q2 · Tốc độ mất giá theo cụm hãng, có và không kiểm soát"),
        "era_rho": lambda: figures.plot_era_correlation(corr[corr["scope"] == "year_era"], fig_dir, version),
        "price_by_condition": lambda: figures.plot_price_by_category(
            df, "condition_clean", fig_dir, version, order=list(figures.CONDITION_ORDER)),
        "price_by_title": lambda: figures.plot_price_by_category(
            df, "title_status_clean", fig_dir, version, order=title_order),
        "price_by_type": lambda: figures.plot_price_by_category(df, "type", fig_dir, version),
        "common_ranking": lambda: figures.plot_common_ranking(t["common_scale_ranking"], fig_dir, version),
    }
    if "confounding_condition_by_seller" in t:
        jobs["condition_by_seller"] = lambda: figures.plot_condition_by_seller(
            t["confounding_condition_by_seller"], fig_dir, version)
    for name, job in jobs.items():
        try:
            log.outputs.append(f"figures/{job().name}")
        except Exception as exc:  # noqa: BLE001 - ghi lại để run trả mã lỗi
            log.step(f"figure:{name}", "FAIL", repr(exc))
    log.step("Hình", "OK", f"{sum(o.startswith('figures/') for o in log.outputs)} hình")


def export_bi(df: pd.DataFrame, lookup: pd.DataFrame, out_dir: Path, version: str, log: RunLog) -> None:
    """Bảng Star Schema cho Power BI và giá trị Python để đối chiếu measure DAX."""
    tables = bi_export.build_star_schema(df, lookup)
    problems = bi_export.validate_star_schema(tables)
    if problems:
        log.step("bi_export", "FAIL", "; ".join(problems))
        return
    bi_dir = out_dir / "bi"
    paths = bi_export.export_tables(tables, bi_dir)
    parity = bi_export.dax_parity_expected(tables["Fact_Listings"])
    parity.insert(0, "dataset_version", version)
    parity.to_csv(bi_dir / "dax_parity_check.csv", index=False, encoding="utf-8-sig")
    log.outputs.extend(f"bi/{p.name}" for p in [*paths, bi_dir / "dax_parity_check.csv"])
    log.step("bi_export", "OK", f"{len(tables['Fact_Listings']):,} dòng Fact, {len(tables) - 1} bảng Dim")


def write_manifest(out: Path, abt: Path, version: str, df: pd.DataFrame, uncovered: list[str],
                   log: RunLog, started_at: str, seconds: float) -> None:
    import matplotlib
    import scipy
    import statsmodels

    manifest = {
        "dataset_version": version, "abt_path": str(abt), "abt_sha256": sha256_of(abt),
        "abt_rows": int(len(df)), "abt_columns": list(map(str, df.columns)), "uncovered_brands": uncovered,
        "started_at": started_at, "seconds": round(seconds, 2), "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
        "steps": log.steps, "outputs": sorted(log.outputs),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--abt", type=Path, default=CONFIG.clean_parquet, help="ABT TV1 bàn giao (Parquet)")
    parser.add_argument("--dataset-version", default=None, help="Ghi vào mọi bảng đầu ra, ví dụ v1.0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--transform-log", type=Path, default=None, help="m1_transform_log.csv cho phễu làm sạch")
    parser.add_argument("--brand-lookup", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-kendall", action="store_true", help="Bỏ Kendall τ (kiểm tra độ bền)")
    parser.add_argument("--bi", action="store_true", help="Xuất bảng Power BI vào <out>/bi")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, started_at = time.perf_counter(), datetime.now(UTC).isoformat(timespec="seconds")
    log, out = RunLog(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    version = args.dataset_version or Path(args.abt).stem
    df = load_abt(args.abt, log)
    lookup = ga.load_brand_lookup(args.brand_lookup)
    df, uncovered = ga.attach_brand_lookup(df, lookup)
    if uncovered:
        log.step("brand_lookup", "WARN", f"hãng chưa có trong bảng tra: {uncovered}")
    log.step("Đọc ABT", "OK", f"{len(df):,} dòng · phiên bản {version}")
    t: dict[str, pd.DataFrame] = {"eda_numeric": eda_summary.numeric_summary(df, dataset_version=version)}
    t["eda_categorical"], t["eda_levels"] = eda_summary.categorical_summary(df, dataset_version=version)
    t["eda_insights"] = eda_summary.insight_sentences(t["eda_numeric"])
    if args.transform_log:
        t["cleaning_funnel"] = eda_summary.cleaning_funnel(pd.read_csv(args.transform_log))
    t["correlation_results"], t["partial_correlations"] = run_correlations(df, version, kendall=not args.no_kendall)
    t["group_depreciation"], t["median_curves"], t["era_ranges"] = run_groups(df, version)
    t["categorical_association"], t["level_summaries"], t["common_scale_ranking"] = run_association(df, version)
    t.update(run_confounding(df, version, log))
    ranking = run_ranking(df, version, log)
    if ranking is not None:
        t["feature_ranking_m2"] = ranking
    t["m2_key_numbers"] = build_key_numbers(version, df, t)
    for name, table in t.items():
        write_table(table, out, name, log)
    log.step("Bảng kết quả", "OK", f"{len(t)} bảng CSV trong {out}")
    if not args.no_figures:
        make_figures(out, version, df, t, log)
    if args.bi:
        export_bi(df, lookup, out, version, log)
    write_manifest(out, Path(args.abt), version, df, uncovered, log, started_at, time.perf_counter() - start)
    print(t["m2_key_numbers"][["key", "value", "unit"]].to_string(index=False), flush=True)
    return 1 if log.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
