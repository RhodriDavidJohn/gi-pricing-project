from types import SimpleNamespace

import pandas as pd

import download_data
from lib.data import load_raw_data


def fake_openml(raw_data, calls):
    df_frq, df_sev = raw_data

    def fetch_openml(data_id, as_frame):
        calls.append(data_id)
        return SimpleNamespace(frame={41214: df_frq, 41215: df_sev}[data_id])

    return fetch_openml


def test_download_saves_both_files(cfg, raw_data, monkeypatch):
    calls = []
    monkeypatch.setattr(download_data, "fetch_openml", fake_openml(raw_data, calls))

    # the fixture already wrote the raw files, so nothing is downloaded...
    download_data.main(cfg)
    assert calls == []

    # ...unless forced
    paths = download_data.main(cfg, force=True)
    assert sorted(calls) == [41214, 41215]

    df_frq, df_sev = load_raw_data(cfg["paths"]["raw_dir"])
    pd.testing.assert_frame_equal(df_frq, raw_data[0], check_dtype=False)
    assert len(df_sev) == len(raw_data[1])
    assert set(paths) == {"freq", "sev"}


def test_download_only_fetches_missing_files(cfg, raw_data, monkeypatch):
    calls = []
    monkeypatch.setattr(download_data, "fetch_openml", fake_openml(raw_data, calls))

    (cfg["paths"]["raw_dir"] / download_data.SEV_FILE).unlink()
    download_data.main(cfg)
    assert calls == [41215]
