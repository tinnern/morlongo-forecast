from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fetch_station_history import download_module


def test_pagination_keeps_bucket_boundaries():
    session = Mock()
    session.get.side_effect = [
        Mock(json=lambda: {"status": "ok", "body": {"900": [10, 60]}}),
        Mock(json=lambda: {"status": "ok", "body": {"2700": [12, 62]}}),
        Mock(json=lambda: {"status": "ok", "body": {}}),
    ]
    with TemporaryDirectory() as directory, \
            patch("fetch_station_history.ARCHIVE", Path(directory)), \
            patch("fetch_station_history.time.sleep"):
        name, readings = download_module(session, {"_id": "station"},
                                         {"_id": "outdoor", "type": "NAModule1"}, 5399)
    assert name == "outdoor"
    assert list(readings) == [900, 2700]
    assert [call.kwargs["params"]["date_begin"] for call in session.get.call_args_list] == [0, 1800, 3600]


if __name__ == "__main__":
    test_pagination_keeps_bucket_boundaries()
    print("History checks passed")
