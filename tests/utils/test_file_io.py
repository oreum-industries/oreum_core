"""Tests for utils.file_io.BaseFileIO and check_fqns_exist"""

from pathlib import Path

import pytest

from oreum_core.utils.file_io import BaseFileIO, check_fqns_exist


class TestBaseFileIOInit:
    """Tests for BaseFileIO.__init__()"""

    def test_init_default_and_valid_rootdir(self, tmp_path):
        """Happy: no rootdir → rootdir set to cwd; valid existing dir → rootdir set correctly"""
        bio_default = BaseFileIO()
        assert bio_default.rootdir == Path.cwd()
        bio_explicit = BaseFileIO(rootdir=tmp_path)
        assert bio_explicit.rootdir == tmp_path

    def test_init_nonexistent_rootdir_creates_it(self, tmp_path):
        """Happy: nonexistent rootdir → auto-created on init"""
        new_dir = tmp_path / "no_such_dir"
        assert not new_dir.exists()
        bio = BaseFileIO(rootdir=new_dir)
        assert new_dir.is_dir()
        assert bio.rootdir == new_dir

    def test_init_nested_nonexistent_rootdir_creates_it(self, tmp_path):
        """Happy: deeply nested nonexistent rootdir → all dirs auto-created"""
        new_dir = tmp_path / "a" / "b" / "c"
        assert not new_dir.exists()
        BaseFileIO(rootdir=new_dir)
        assert new_dir.is_dir()


class TestBaseFileIOGetPathRead:
    """Tests for BaseFileIO.get_path_read()"""

    def test_existing_file_returns_path(self, tmp_path):
        """Happy: file exists → returns correct Path"""
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        bio = BaseFileIO(rootdir=tmp_path)
        assert bio.get_path_read("data.csv") == f

    def test_accepts_path_input(self, tmp_path):
        """Happy: fn as Path object → returns correct Path"""
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        bio = BaseFileIO(rootdir=tmp_path)
        assert bio.get_path_read(Path("data.csv")) == f

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

    def test_accepts_path_input(self, tmp_path):
        """Happy: fn as Path object → returns correct fqn"""
        bio = BaseFileIO(rootdir=tmp_path)
        result = bio.get_path_write(Path("output.csv"))
        assert result == tmp_path / "output.csv"

    def test_missing_subdir_is_created(self, tmp_path):
        """Happy: parent subdir doesn't exist → auto-created, returns fqn"""
        bio = BaseFileIO(rootdir=tmp_path)
        result = bio.get_path_write("new_dir/output.csv")
        assert result == tmp_path / "new_dir" / "output.csv"
        assert (tmp_path / "new_dir").is_dir()

    def test_nested_missing_subdir_is_created(self, tmp_path):
        """Happy: deeply nested missing subdirs → all created, returns fqn"""
        bio = BaseFileIO(rootdir=tmp_path)
        result = bio.get_path_write("a/b/c/output.csv")
        assert result == tmp_path / "a" / "b" / "c" / "output.csv"
        assert (tmp_path / "a" / "b" / "c").is_dir()


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
