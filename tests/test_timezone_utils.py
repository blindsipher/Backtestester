import pandas as pd
import pytest
from utils import timezone_utils as tz


def test_convert_to_target_timezone_naive_series():
    series = pd.Series(pd.date_range("2024-01-01", periods=2, freq="H"))
    converted = tz.convert_to_target_timezone(series)
    assert str(converted.dt.tz) == tz.TARGET_TIMEZONE


def test_convert_datetime_index_to_column():
    index = pd.date_range("2024-01-01", periods=2, freq="H", tz="UTC")
    df = pd.DataFrame({"value": [1, 2]}, index=index)
    result = tz.convert_datetime_index_to_column(df)
    assert "datetime" in result.columns
    assert str(result["datetime"].dt.tz) == tz.TARGET_TIMEZONE


def test_ensure_datetime_column_creates_from_candidates():
    df = pd.DataFrame({"timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"]})
    result = tz.ensure_datetime_column(df)
    assert "datetime" in result.columns
    assert str(result["datetime"].dt.tz) == tz.TARGET_TIMEZONE


def test_validate_datetime_column_success():
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=1, tz=tz.TARGET_TIMEZONE)})
    assert tz.validate_datetime_column(df)


def test_validate_datetime_column_missing_raises():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        tz.validate_datetime_column(df)
