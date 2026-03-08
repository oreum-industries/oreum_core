"""Shared fixtures for eda tests"""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to prevent memory accumulation"""
    yield
    plt.close("all")
