import time
from typing import Optional

from src.core.consts import FORMATS


def create_run_id(run_id: Optional[str]) -> str:
    return run_id if (run_id is not None) else time.strftime(FORMATS.TIME)
