"""Current franchise identities in model inputs and published teams."""

import pandas as pd

from backend import publish
from backend.etl import store


def test_published_names_and_model_inputs_use_current_franchises(monkeypatch):
    raw = pd.DataFrame(
        [
            ("LA", "Los Angeles Rams"),
            ("LAC", "Los Angeles Chargers"),
            ("LV", "Las Vegas Raiders"),
            ("LA", "St. Louis Rams"),
            ("LAC", "San Diego Chargers"),
            ("LV", "Oakland Raiders"),
        ],
        columns=["team_abbr", "team_name"],
    ).assign(team_color="#000000", team_color2="#ffffff", team_logo_espn="logo")
    monkeypatch.setattr(store, "read_raw", lambda *parts: raw)
    expected = {
        "LA": "Los Angeles Rams",
        "LAC": "Los Angeles Chargers",
        "LV": "Las Vegas Raiders",
    }
    assert store.team_names() == expected
    published = publish.build_teams_frame().set_index("team_abbr")["team"].to_dict()
    assert published == expected
