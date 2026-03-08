"""Tests for eda.eda_io.FigureIO"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib import figure

from oreum_core.eda.eda_io import FigureIO


@pytest.fixture
def simple_fig():
    """Minimal matplotlib Figure for tests"""
    with plt.ioff():
        f, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
    yield f
    plt.close(f)


class TestFigureIOWrite:
    """Tests for FigureIO.write()"""

    def test_write_returns_path(self, tmp_path, simple_fig):
        """Happy: write returns a Path"""
        io = FigureIO(rootdir=tmp_path)
        result = io.write(simple_fig, "myplot")
        assert isinstance(result, Path)

    def test_write_suffix_is_png(self, tmp_path, simple_fig):
        """Happy: written file has .png suffix"""
        io = FigureIO(rootdir=tmp_path)
        result = io.write(simple_fig, "myplot")
        assert result.suffix == ".png"

    def test_write_file_exists_on_disk(self, tmp_path, simple_fig):
        """Happy: written file actually exists"""
        io = FigureIO(rootdir=tmp_path)
        result = io.write(simple_fig, "myplot")
        assert result.exists()

    def test_write_fn_with_dots_gets_png_suffix(self, tmp_path, simple_fig):
        """Happy: fn with embedded dots → .png suffix applied correctly"""
        io = FigureIO(rootdir=tmp_path)
        result = io.write(simple_fig, "plot.v2.final")
        assert result.suffix == ".png"
        assert result.exists()


class TestFigureIORead:
    """Tests for FigureIO.read()"""

    def test_read_returns_figure(self, tmp_path, simple_fig):
        """Happy: read existing PNG → returns matplotlib Figure"""
        io = FigureIO(rootdir=tmp_path)
        io.write(simple_fig, "myplot")
        result = io.read(fn="myplot")
        assert isinstance(result, figure.Figure)
        plt.close(result)

    def test_read_no_fn_no_fqn_raises(self, tmp_path):
        """Sad: neither fn nor fqn supplied → ValueError"""
        io = FigureIO(rootdir=tmp_path)
        with pytest.raises(ValueError, match="fqn or fn"):
            io.read()
