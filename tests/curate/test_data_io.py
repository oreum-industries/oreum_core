"""Tests for curate.data_io I/O classes"""

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest

from oreum_core.curate.data_io import (
    DaskParquetIO,
    PandasCSVIO,
    PandasExcelIO,
    PandasParquetIO,
    PickleIO,
    SimpleStringIO,
)


@pytest.fixture
def simple_df():
    """Minimal DataFrame for round-trip tests"""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


class TestPandasCSVIO:
    """Tests for PandasCSVIO read/write"""

    def test_write_returns_path_and_file_exists(self, tmp_path, simple_df):
        """Happy: write returns Path with .csv suffix and file exists on disk"""
        io = PandasCSVIO(rootdir=tmp_path)
        result = io.write(simple_df, "test")
        assert isinstance(result, Path)
        assert result.suffix == ".csv"
        assert result.exists()

    def test_roundtrip(self, tmp_path, simple_df):
        """Happy: write then read returns equivalent data columns"""
        io = PandasCSVIO(rootdir=tmp_path)
        io.write(simple_df, "test")
        df_read = io.read("test")
        pd.testing.assert_frame_equal(df_read[["a", "b"]], simple_df, check_dtype=False)

    def test_read_adds_csv_extension(self, tmp_path, simple_df):
        """Happy: read accepts fn without extension"""
        io = PandasCSVIO(rootdir=tmp_path)
        io.write(simple_df, "myfile")
        df_read = io.read("myfile")
        assert "a" in df_read.columns

    def test_read_missing_file_raises(self, tmp_path):
        """Sad: missing file → FileNotFoundError"""
        io = PandasCSVIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            io.read("nonexistent")


class TestPandasParquetIO:
    """Tests for PandasParquetIO read/write"""

    def test_write_returns_path_and_file_exists(self, tmp_path, simple_df):
        """Happy: write returns Path with .parquet suffix and file exists on disk"""
        io = PandasParquetIO(rootdir=tmp_path)
        result = io.write(simple_df, "test")
        assert isinstance(result, Path)
        assert result.suffix == ".parquet"
        assert result.exists()

    def test_roundtrip(self, tmp_path, simple_df):
        """Happy: write then read returns equivalent DataFrame"""
        io = PandasParquetIO(rootdir=tmp_path)
        io.write(simple_df, "test")
        df_read = io.read("test")
        pd.testing.assert_frame_equal(df_read, simple_df)

    def test_read_missing_file_raises(self, tmp_path):
        """Sad: missing file → FileNotFoundError"""
        io = PandasParquetIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            io.read("nonexistent")


class TestPickleIO:
    """Tests for PickleIO read/write"""

    def test_init_invalid_kind_raises(self, tmp_path):
        """Sad: invalid kind → ValueError"""
        with pytest.raises(ValueError, match="kind must be"):
            PickleIO(kind="text", rootdir=tmp_path)

    def test_init_valid_kind(self, tmp_path):
        """Happy: kind='bytes' initialises without error"""
        io = PickleIO(kind="bytes", rootdir=tmp_path)
        assert io.k == "bytes"

    def test_write_returns_path(self, tmp_path):
        """Happy: write returns a Path with .pickle suffix"""
        io = PickleIO(rootdir=tmp_path)
        result = io.write({"k": "v"}, "myobj")
        assert isinstance(result, Path)
        assert result.suffix == ".pickle"

    def test_roundtrip_dict(self, tmp_path):
        """Happy: write dict, read back equals original"""
        io = PickleIO(rootdir=tmp_path)
        obj = {"x": [1, 2, 3], "y": "hello"}
        io.write(obj, "myobj")
        assert io.read("myobj") == obj

    def test_read_missing_file_raises(self, tmp_path):
        """Sad: missing file → FileNotFoundError"""
        io = PickleIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            io.read("ghost")


class TestSimpleStringIO:
    """Tests for SimpleStringIO read/write"""

    def test_init_invalid_kind_raises(self, tmp_path):
        """Sad: invalid kind → ValueError"""
        with pytest.raises(ValueError, match="kind must be"):
            SimpleStringIO(kind="xml", rootdir=tmp_path)

    def test_txt_write_returns_path(self, tmp_path):
        """Happy: txt write returns a Path with .txt suffix"""
        io = SimpleStringIO(kind="txt", rootdir=tmp_path)
        result = io.write("hello", "note")
        assert isinstance(result, Path)
        assert result.suffix == ".txt"

    def test_txt_roundtrip(self, tmp_path):
        """Happy: txt write then read returns original string"""
        io = SimpleStringIO(kind="txt", rootdir=tmp_path)
        io.write("hello world", "note")
        assert io.read("note.txt") == "hello world"

    def test_json_roundtrip(self, tmp_path):
        """Happy: json write then read returns original dict"""
        io = SimpleStringIO(kind="json", rootdir=tmp_path)
        data = {"key": "value", "n": 42}
        io.write(data, "mydata")
        assert io.read("mydata.json") == data

    def test_read_missing_file_raises(self, tmp_path):
        """Sad: missing file → FileNotFoundError"""
        io = SimpleStringIO(kind="txt", rootdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            io.read("ghost.txt")


class TestDaskParquetIO:
    """Tests for DaskParquetIO"""

    def test_write_raises_not_implemented(self, tmp_path, simple_df):
        """Sad: write raises NotImplementedError (not yet implemented)"""
        io = DaskParquetIO(rootdir=tmp_path)
        ddf = dd.from_pandas(simple_df)
        with pytest.raises(NotImplementedError):
            io.write(ddf, "out")

    def test_read_missing_file_raises(self, tmp_path):
        """Sad: missing parquet file → FileNotFoundError"""
        io = DaskParquetIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            io.read("nonexistent")


class TestPandasExcelIO:
    """Tests for PandasExcelIO read/write and writer workflow"""

    def test_write_returns_path_and_file_exists(self, tmp_path, simple_df):
        """Happy: write returns Path with .xlsx suffix and file exists on disk"""
        io = PandasExcelIO(rootdir=tmp_path)
        result = io.write(simple_df, "report", index=False)
        assert isinstance(result, Path)
        assert result.suffix == ".xlsx"
        assert result.exists()

    def test_write_roundtrip(self, tmp_path, simple_df):
        """Happy: write then read returns equivalent DataFrame"""
        io = PandasExcelIO(rootdir=tmp_path)
        io.write(simple_df, "report", index=False)
        df_read = io.read("report")
        pd.testing.assert_frame_equal(df_read, simple_df, check_dtype=False)

    def test_writer_write_before_open_raises(self, tmp_path, simple_df):
        """Sad: writer_write before writer_open → RuntimeError"""
        io = PandasExcelIO(rootdir=tmp_path)
        with pytest.raises(RuntimeError, match="writer_open"):
            io.writer_write(simple_df)

    def test_writer_close_before_open_raises(self, tmp_path):
        """Sad: writer_close before writer_open → RuntimeError"""
        io = PandasExcelIO(rootdir=tmp_path)
        with pytest.raises(RuntimeError, match="writer_open"):
            io.writer_close()

    def test_multi_sheet_workflow_returns_path(self, tmp_path, simple_df):
        """Happy: open → write two sheets → close returns Path"""
        io = PandasExcelIO(rootdir=tmp_path)
        io.writer_open("multi")
        io.writer_write(simple_df, sheet_name="s1", index=False)
        io.writer_write(simple_df, sheet_name="s2", index=False)
        result = io.writer_close()
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".xlsx"
