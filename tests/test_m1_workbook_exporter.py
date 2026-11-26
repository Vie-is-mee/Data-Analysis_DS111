"""Kiểm thử xuất Workbook TV1: đủ sheet, giữ sheet TV3, Sheet 07 đối chiếu đúng."""

from openpyxl import load_workbook

from src.data.pipeline_m1 import build_clean_dataset
from src.data.raw_profile import load_raw, profile_raw
from src.data.workbook_exporter import M1_SHEETS, SAMPLE_SIZE, export_workbook
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
