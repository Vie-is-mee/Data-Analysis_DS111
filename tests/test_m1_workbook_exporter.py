"""Kiểm thử xuất Workbook TV1: đủ sheet, giữ sheet TV3, Sheet 07 đối chiếu đúng."""

from openpyxl import load_workbook

from src.data.pipeline_m1 import build_clean_dataset
from src.data.raw_profile import load_raw, profile_raw
from src.data.workbook_exporter import M1_SHEETS, SAMPLE_SIZE, build_label_table, export_workbook
from tests.test_m1_pipeline_smoke import _fake_raw


def _export(tmp_path, out):
    raw_csv = tmp_path / "vehicles.csv"
    _fake_raw().to_csv(raw_csv, index=False)
    raw = load_raw(raw_csv)
    result = build_clean_dataset(raw)
    export_workbook(result.data, result.log.to_dataframe(), result.store,
                    profile_raw(load_raw(raw_csv, usecols=None)), raw, out)
    return result


def test_exporter_writes_all_sheets_and_keeps_m3_sheets(tmp_path):
    out = tmp_path / "Project_Workbook.xlsx"
    _export(tmp_path, out)
    book = load_workbook(out)
    assert list(book.sheetnames[:8]) == list(M1_SHEETS)
    assert "08_Experiment_Tracking" in book.sheetnames and len(book.sheetnames) == 9

    # TV3 ghi sheet 08 → TV1 xuất lại không được làm mất.
    book["08_Experiment_Tracking"]["A1"] = "Exp0 R2=0.5"
    book.create_sheet("08b_VIF_Feature_Ranking")["A1"] = "giữ nguyên"
    book.save(out)
    _export(tmp_path, out)
    book = load_workbook(out)
    assert book["08_Experiment_Tracking"]["A1"].value == "Exp0 R2=0.5"
    assert book["08b_VIF_Feature_Ranking"]["A1"].value == "giữ nguyên"
    assert book.sheetnames.count("00_Log") == 1


def test_sheet07_compares_fixed_ids_one_to_one(tmp_path):
    out = tmp_path / "wb.xlsx"
    result = _export(tmp_path, out)
    ws = load_workbook(out)["07_Sample_Comparison"]
    header = [c.value for c in ws[2]]
    rows = list(ws.iter_rows(min_row=3, values_only=True))
    assert header[:2] == ["listing_id", "Trường đã thay đổi"]
    assert len(rows) == SAMPLE_SIZE
    ids = [r[0] for r in rows]
    assert len(set(ids)) == SAMPLE_SIZE and set(ids) <= set(result.data["listing_id"])
    # Dữ liệu giả có 'Chevy', 'VW', nhãn lạ → phải có dòng ghi nhận thay đổi.
    assert any(r[1] != "—" for r in rows)

    # Chạy lại cho cùng mẫu (seed cố định).
    _export(tmp_path, out)
    again = [r[0] for r in load_workbook(out)["07_Sample_Comparison"].iter_rows(min_row=3, values_only=True)]
    assert again == ids


def test_label_table_covers_every_label_with_ordinal_codes(tmp_path):
    out = tmp_path / "Project_Workbook.xlsx"
    data = _export(tmp_path, out).data
    table = build_label_table(data)

    # Mỗi cột: đủ mọi nhãn, tổng số dòng = số dòng ABT.
    for col, part in table.groupby("Cột"):
        assert set(part["Nhãn chuẩn"]) == set(data[col].unique())
        assert part["Số dòng"].sum() == len(data)

    cond = table[table["Cột"] == "condition_clean"].set_index("Nhãn chuẩn")
    assert cond["Mã Ordinal"].drop("unknown", errors="ignore").is_monotonic_increasing
    if "unknown" in cond.index:
        assert cond.loc["unknown", "Mã Ordinal"] == ""
    # Luật gộp lấy thẳng từ code: chevy/chev → chevrolet.
    manu = table[table["Cột"] == "manufacturer_clean"].set_index("Nhãn chuẩn")
    if "chevrolet" in manu.index:
        assert "chevy" in manu.loc["chevrolet", "Gộp từ nhãn gốc"]

    sheet = load_workbook(out)["05_Encoding"]
    texts = [c.value for row in sheet.iter_rows() for c in row if isinstance(c.value, str)]
    assert any(t.startswith("Bảng nhãn chuẩn") for t in texts)


def test_selection_table_covers_every_column_with_train_only_evidence(tmp_path):
    from src.data.selector import DROP, KEEP, LINEAR, TARGET, TREE, TRY, selection_evidence

    out = tmp_path / "Project_Workbook.xlsx"
    data = _export(tmp_path, out).data
    table, vif, notes = selection_evidence(data)

    # Mỗi cột ABT có đúng một dòng quyết định, mỗi quyết định thuộc miền cố định.
    assert sorted(table["Cột"]) == sorted(data.columns)
    assert set(table[LINEAR]) | set(table[TREE]) <= {KEEP, DROP, TRY, TARGET}
    assert table["Bằng chứng định lượng"].str.len().gt(0).all()
    pick = table.set_index("Cột")
    assert pick.loc["year", LINEAR] == DROP            # posting_year hằng → year trùng car_age
    assert pick.loc["posting_year", LINEAR] == DROP
    assert pick.loc["price_outlier_iqr", LINEAR] == DROP  # tính từ price → rò rỉ
    assert pick.loc["car_age", LINEAR] == KEEP
    assert pick.loc["car_age_squared", TREE] == DROP   # biến đổi đơn điệu: thừa với cây
    assert set(vif["Bộ biến"].str[:2]) == {"1.", "2.", "3."} and len(notes) >= 3

    # Căn cứ chọn biến chỉ dùng train: đổi giá val/test không làm đổi số liệu (trừ split_set, đo toàn ABT).
    noisy = data.copy()
    other = noisy["split_set"].ne("train")
    noisy.loc[other, "log_price"] = noisy.loc[other, "log_price"].sample(frac=1, random_state=1).to_numpy()
    again, _, _ = selection_evidence(noisy)
    keep_rows = table["Cột"].ne("split_set")
    assert again.loc[keep_rows, "Giá trị"].equals(table.loc[keep_rows, "Giá trị"])

    sheet = load_workbook(out)["06_Selection"]
    texts = [c.value for row in sheet.iter_rows() for c in row if isinstance(c.value, str)]
    assert any(t.startswith("Quyết định chọn biến") for t in texts)
    assert any(t.startswith("• Bộ 2: car_age và car_age²") for t in texts)
