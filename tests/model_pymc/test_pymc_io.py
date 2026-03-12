"""Tests for model_pymc.pymc_io.PYMCIO"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import arviz as az

    from oreum_core.model_pymc.pymc_io import PYMCIO

    HAS_PYMC = True
except Exception:
    PYMCIO = None  # type: ignore[assignment,misc]
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")


@pytest.fixture
def tmp_pymcio(tmp_path):
    """PYMCIO instance with a temporary rootdir"""
    return PYMCIO(rootdir=tmp_path)


@pytest.fixture
def simple_idata():
    """Minimal ArviZ InferenceData for round-trip tests"""
    rng = np.random.default_rng(42)
    return az.from_dict(posterior={"alpha": rng.normal(size=(4, 100))})


@pytest.fixture
def mock_mdl():
    """MagicMock with minimal BasePYMCModel interface"""
    mdl = MagicMock()
    mdl.mdl_id_dn = "test_model_v0"
    mdl.mdl_id_fn = "test_model_v0_obs"
    return mdl


class TestPYMCIOWriteIdata:
    """Tests for PYMCIO.write_idata()"""

    def test_write_returns_path_and_file_exists(
        self, tmp_pymcio, mock_mdl, simple_idata
    ):
        """Happy: write_idata with explicit fn → file in mdl_id_dn subdir with .netcdf suffix"""
        result = tmp_pymcio.write_idata(mock_mdl, idata=simple_idata, fn="myout")
        assert isinstance(result, Path)
        assert result.suffix == ".netcdf"
        assert result.exists()
        assert result.parent.name == mock_mdl.mdl_id_dn

    def test_write_uses_mdl_id_fn_when_no_fn(self, tmp_pymcio, mock_mdl, simple_idata):
        """Happy: fn='' → filename derived from mdl.mdl_id_fn, subdir from mdl.mdl_id_dn"""
        result = tmp_pymcio.write_idata(mock_mdl, idata=simple_idata)
        assert mock_mdl.mdl_id_fn in result.name
        assert result.parent.name == mock_mdl.mdl_id_dn

    def test_write_creates_subdir(self, tmp_pymcio, mock_mdl, simple_idata):
        """Happy: mdl_id_dn subdir doesn't exist → auto-created on write"""
        subdir = tmp_pymcio.rootdir / mock_mdl.mdl_id_dn
        assert not subdir.exists()
        tmp_pymcio.write_idata(mock_mdl, idata=simple_idata)
        assert subdir.is_dir()

    def test_write_cleans_fn(self, tmp_pymcio, mock_mdl, simple_idata):
        """Happy: dirty fn → snl.clean applied, filename is sanitised"""
        result = tmp_pymcio.write_idata(mock_mdl, idata=simple_idata, fn="My Output!")
        assert result.name == "my_output.netcdf"

    def test_write_uses_mdl_idata_when_idata_is_none(
        self, tmp_pymcio, mock_mdl, simple_idata
    ):
        """Happy: idata=None → mdl.idata.to_netcdf() used"""
        mock_mdl.idata = simple_idata
        result = tmp_pymcio.write_idata(mock_mdl, fn="via_mdl")
        assert result.exists()


class TestPYMCIOReadIdata:
    """Tests for PYMCIO.read_idata()"""

    def test_roundtrip_via_fn(self, tmp_pymcio, simple_idata):
        """Happy: file at flat rootdir path read via fn → returns InferenceData"""
        flat_fqn = tmp_pymcio.rootdir / "myidata.netcdf"
        simple_idata.to_netcdf(str(flat_fqn))
        result = tmp_pymcio.read_idata(fn="myidata")
        assert isinstance(result, az.InferenceData)
        assert "posterior" in result

    def test_roundtrip_via_mdl(self, tmp_pymcio, mock_mdl, simple_idata):
        """Happy: write then read via mdl → round-trip through mdl_id_dn subdir"""
        tmp_pymcio.write_idata(mock_mdl, idata=simple_idata)
        result = tmp_pymcio.read_idata(mdl=mock_mdl)
        assert isinstance(result, az.InferenceData)
        assert "posterior" in result

    def test_read_with_mdl_uses_subdir(self, tmp_pymcio, mock_mdl, simple_idata):
        """Happy: read_idata(mdl=...) looks inside mdl_id_dn subdir"""
        tmp_pymcio.write_idata(mock_mdl, idata=simple_idata)
        subdir = tmp_pymcio.rootdir / mock_mdl.mdl_id_dn
        assert subdir.is_dir()
        result = tmp_pymcio.read_idata(mdl=mock_mdl)
        assert isinstance(result, az.InferenceData)

    def test_read_cleans_fn(self, tmp_pymcio, simple_idata):
        """Happy: dirty fn → snl.clean applied before path lookup"""
        flat_fqn = tmp_pymcio.rootdir / "my_file.netcdf"
        simple_idata.to_netcdf(str(flat_fqn))
        result = tmp_pymcio.read_idata(fn="My File!")
        assert isinstance(result, az.InferenceData)

    def test_read_missing_file_raises(self, tmp_pymcio):
        """Sad: file not found → FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            tmp_pymcio.read_idata(fn="ghost")


class TestPYMCIOWriteGraph:
    """Tests for PYMCIO.write_graph()"""

    def test_write_graph_no_write_returns_graphviz(self, tmp_pymcio, mock_mdl):
        """Happy: write=False → returns graphviz object directly"""
        mock_gv = MagicMock()
        with patch(
            "oreum_core.model_pymc.pymc_io.model_to_graphviz", return_value=mock_gv
        ):
            result = tmp_pymcio.write_graph(mock_mdl, fn="mygraph", write=False)
        assert result is mock_gv

    def test_write_graph_creates_subdir(self, tmp_pymcio, mock_mdl):
        """Happy: write_graph → mdl_id_dn subdir auto-created (even when write=False)"""
        mock_gv = MagicMock()
        subdir = tmp_pymcio.rootdir / mock_mdl.mdl_id_dn
        assert not subdir.exists()
        with patch(
            "oreum_core.model_pymc.pymc_io.model_to_graphviz", return_value=mock_gv
        ):
            tmp_pymcio.write_graph(mock_mdl, fn="mygraph", write=False)
        assert subdir.is_dir()

    def test_write_graph_path_uses_subdir(self, tmp_pymcio, mock_mdl):
        """Happy: write_graph with write=True → fqn parent is mdl_id_dn subdir"""
        mock_gv = MagicMock()
        with patch(
            "oreum_core.model_pymc.pymc_io.model_to_graphviz", return_value=mock_gv
        ):
            result = tmp_pymcio.write_graph(mock_mdl, fn="mygraph")
        assert result.parent.name == mock_mdl.mdl_id_dn

    def test_write_graph_cleans_fn(self, tmp_pymcio, mock_mdl):
        """Happy: dirty fn → snl.clean applied, returned fqn stem is sanitised"""
        mock_gv = MagicMock()
        with patch(
            "oreum_core.model_pymc.pymc_io.model_to_graphviz", return_value=mock_gv
        ):
            result = tmp_pymcio.write_graph(mock_mdl, fn="My Graph!")
        assert result.name == "my_graph.png"

    def test_write_graph_invalid_format_raises(self, tmp_pymcio, mock_mdl):
        """Sad: unsupported format → ValueError"""
        mock_gv = MagicMock()
        with patch(
            "oreum_core.model_pymc.pymc_io.model_to_graphviz", return_value=mock_gv
        ):
            with pytest.raises(ValueError, match="png"):
                tmp_pymcio.write_graph(mock_mdl, fn="mygraph", fmt="pdf")
