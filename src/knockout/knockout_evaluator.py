from typing import Iterable, Tuple

import pandas as pd

from .knockout_mode import KnockoutMode


class KnockoutEvaluator(object):
    def knockout_eval(
        self, dataset: pd.DataFrame, layers: Iterable[int], knockout_mode: KnockoutMode
    ) -> Tuple[pd.DataFrame, int]:
        raise NotImplementedError
