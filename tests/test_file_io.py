"""Tests for utils.file_io.BaseFileIO and check_fqns_exist"""

from pathlib import Path

import pytest

from oreum_core.utils.file_io import BaseFileIO, check_fqns_exist


class TestBaseFileIOInit:
    """Tests for BaseFileIO.__init__()"""

    def test_init_default_rootdir_is_cwd(self):
        """Happy: no rootdir → rootdir set to cwd"""
        bio = BaseFileIO()
        assert bio.rootdir == Path.cwd()

    def test_init_valid_rootdir(self, tmp_path):
        """Happy: valid existing dir → rootdir set correctly"""
        bio = BaseFileIO(rootdir=tmp_path)
        assert bio.rootdir == tmp_path

    def test_init_nonexistent_rootdir_raises(self, tmp_path):
        """Sad: nonexistent dir → FileNotFoundError"""
        bad = tmp_path / "no_such_dir"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            BaseFileIO(rootdir=bad)


class TestBaseFileIOGetPathRead:
    """Tests for BaseFileIO.get_path_read()"""

    def test_existing_file_returns_path(self, tmp_path):
        """Happy: file exists → returns correct Path"""
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        bio = BaseFileIO(rootdir=tmp_path)
        assert bio.get_path_read("data.csv") == f

    def test_missing_file_raises(self, tmp_path):
        """Sad: file does not exist → FileNotFoundError"""
        bio = BaseFileIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            bio.get_path_read("ghost.csv")


class TestBaseFileIOGetPathWrite:
    """Tests for BaseFileIO.get_path_write()"""

    def test_valid_dir_returns_path(self, tmp_path):
        """Happy: parent dir exists → returns correct fqn"""
        bio = BaseFileIO(rootdir=tmp_path)
        result = bio.get_path_write("output.csv")
        assert result == tmp_path / "output.csv"

    def test_missing_subdir_raises(self, tmp_path):
        """Sad: parent subdir doesn't exist → FileNotFoundError"""
        bio = BaseFileIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            bio.get_path_write("no_such_dir/output.csv")


class TestCheckFqnsExist:
    """Tests for check_fqns_exist()"""

    def test_all_exist_returns_true(self, tmp_path):
        """Happy: all paths exist → returns True"""
        f1, f2 = tmp_path / "a.txt", tmp_path / "b.txt"
        f1.write_text("x")
        f2.write_text("y")
        assert check_fqns_exist({"a": f1, "b": f2}) is True

    def test_one_missing_raises(self, tmp_path):
        """Sad: one path missing → FileNotFoundError"""
        f1 = tmp_path / "exists.txt"
        f1.write_text("x")
        missing = tmp_path / "ghost.txt"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            check_fqns_exist({"ok": f1, "bad": missing})

    def test_empty_dict_returns_true(self):
        """Edge: empty dict → no files to check, returns True"""
        assert check_fqns_exist({}) is True
