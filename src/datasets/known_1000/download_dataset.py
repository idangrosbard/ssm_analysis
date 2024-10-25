from pathlib import Path

import pandas as pd
import wget


def load_knowns() -> pd.DataFrame:
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    return knowns_df
