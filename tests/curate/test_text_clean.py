"""Tests for curate.text_clean.TextCleaner"""

import numpy as np
import pytest

from oreum_core.curate.text_clean import TextCleaner


@pytest.fixture
def tc():
    """Default TextCleaner instance"""
    return TextCleaner()


class TestBasicClean:
    """Happy-path tests for TextCleaner.basic_clean()"""

    def test_removes_email_address(self, tc):
        """Happy: email address is stripped from text"""
        out = tc.basic_clean("contact user@example.com for help")
        assert "@" not in out

    def test_removes_web_url(self, tc):
        """Happy: www URL is stripped from text"""
        out = tc.basic_clean("visit www.example.com for info")
        assert "www" not in out

    def test_removes_numbers(self, tc):
        """Happy: standalone number tokens are stripped"""
        out = tc.basic_clean("there are 3 cats and 2.5 dogs")
        assert "3" not in out
        assert "2.5" not in out

    def test_removes_nbsp_entity(self, tc):
        """Happy: HTML &nbsp; entity is stripped"""
        out = tc.basic_clean("hello&nbsp;world")
        assert "&nbsp;" not in out

    def test_removes_html_comment(self, tc):
        """Happy: HTML comment block is stripped"""
        out = tc.basic_clean("text <!--embedded css--> content")
        assert "<!--" not in out
        assert "css" not in out

    def test_collapses_four_or_more_repeated_chars(self, tc):
        """Edge: four identical consecutive chars are collapsed to empty"""
        out = tc.basic_clean("aaaa")
        assert out.strip() == ""


class TestConvertBadNumberRepresentation:
    """Tests for TextCleaner.convert_bad_number_representation_to_float()"""

    def test_millions_uppercase(self, tc):
        """Happy: '1M' → 1_000_000.0"""
        assert tc.convert_bad_number_representation_to_float("1M") == pytest.approx(1e6)

    def test_millions_decimal(self, tc):
        """Happy: '1.4m' → 1_400_000.0"""
        assert tc.convert_bad_number_representation_to_float("1.4m") == pytest.approx(
            1.4e6
        )

    def test_thousands(self, tc):
        """Happy: '25K' → 25_000.0"""
        assert tc.convert_bad_number_representation_to_float("25K") == pytest.approx(
            25000.0
        )

    def test_currency_symbol_and_comma(self, tc):
        """Happy: '$3,000.82' → 3000.82 (strips junk chars)"""
        assert tc.convert_bad_number_representation_to_float(
            "$3,000.82"
        ) == pytest.approx(3000.82)

    def test_unrecognised_string_returns_nan(self, tc):
        """Sad: non-numeric string → np.nan"""
        result = tc.convert_bad_number_representation_to_float("xyz")
        assert np.isnan(result)
