"""
Unit tests for the mvml_assignment package and its modules.
"""

from src import __version__
from src.data_cleaner import data_cleaner
from src.get_metrics import produce_report
from src.training import train
from src.inference import inference


def test_version():
    """
    Unit test for version.
    """
    assert __version__ == '0.1.0'


def test_callables():
    """
    Unit test for required callables.
    """
    assert callable(data_cleaner)
    assert callable(produce_report)
    assert callable(train)
    assert callable(inference)
