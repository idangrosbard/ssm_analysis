import pandas as pd
from annotated_text import annotated_text, annotation

from src.consts import COLUMNS
from src.utils.logits import Prompt


def show_prompt(prompt: Prompt):
    annotated_text(
        [
            annotation(val.format(""), col)
            for col in [
                COLUMNS.COUNTER_FACT_COLS.RELATION_PREFIX,
                COLUMNS.COUNTER_FACT_COLS.SUBJECT,
                COLUMNS.COUNTER_FACT_COLS.RELATION_SUFFIX,
                COLUMNS.COUNTER_FACT_COLS.TARGET_TRUE,
            ]
            if pd.notna(val := prompt.get_column(col))
        ]
    )
