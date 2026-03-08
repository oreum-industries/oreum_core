"""Tests for utils.snakey_lowercaser.SnakeyLowercaser"""

import pytest

from oreum_core.utils.snakey_lowercaser import SnakeyLowercaser


@pytest.fixture
def snl():
    """Default SnakeyLowercaser (only underscore preserved)"""
    return SnakeyLowercaser()


@pytest.fixture
def snl_hyphen():
    """SnakeyLowercaser with hyphen also preserved"""
    return SnakeyLowercaser(allowed_punct="-")


class TestClean:
    """Tests for SnakeyLowercaser.clean()"""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("Hello World", "hello_world"),  # spaces joined with underscore
            ("path/to.file", "path_to_file"),  # slashes and dots converted
            ("hello__world", "hello_world"),  # multiple underscores collapsed
            ("_hello_", "hello"),  # leading/trailing underscores stripped
            ("price ($)", "price"),  # punctuation removed
        ],
    )
    def test_clean_happy(self, snl, input_str, expected):
        """Happy path: various inputs produce correct snake_case output"""
        assert snl.clean(input_str) == expected

    def test_clean_empty_string(self, snl):
        """Edge: empty string returns empty string"""
        assert snl.clean("") == ""

    def test_clean_non_string_coerced(self, snl):
        """Edge: non-string input is coerced via str()"""
        assert snl.clean(42) == "42"

    def test_clean_allowed_punct_preserved(self, snl_hyphen):
        """Happy: hyphen is preserved when passed as allowed_punct"""
        assert snl_hyphen.clean("hello-world") == "hello-world"


class TestCleanPatsy:
    """Tests for SnakeyLowercaser.clean_patsy()"""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("sex[T.Male]", "sex_t_male"),  # patsy factor encoding
            ("age:sex", "age_x_sex"),  # patsy interaction term
            ("np.log(income)", "income_log"),  # numpy transform
            ("simple_feature", "simple_feature"),  # plain name passes through
        ],
    )
    def test_clean_patsy_happy(self, snl, input_str, expected):
        """Happy path: patsy-specific transforms produce correct output"""
        assert snl.clean_patsy(input_str) == expected
