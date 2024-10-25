import os
from pathlib import Path
from typing import NamedTuple


class PATHS:
    PROJECT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = PROJECT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'
    DATA_SHARED_DIR = PROJECT_DIR / 'shared'
    RUNS_DIR = PROJECT_DIR / 'runs'
    TENSORBOARD_DIR = PROJECT_DIR / 'tensorboard'


class ENV_VARS:
    MASTER_PORT = 'MASTER_PORT'
    MASTER_ADDR = 'MASTER_ADDR'


class FORMATS:
    TIME = "%Y%m%d_%H-%M-%S"
    LOGGER_FORMAT = '%(asctime)s - %(message)s'


class SPLIT_SIZES:
    DEV = 0.2
    VAL = 0.4
    TEST = 0.4


class DDP:
    MASTER_PORT = os.environ.get(ENV_VARS.MASTER_PORT, '12355')
    MASTER_ADDR = 'localhost'
    BACKEND = 'nccl'
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


class ISlurmArgs(NamedTuple):
    with_parallel: bool
    partition: str = 'gpu-a100-killable'
    time: int = 1200
    singal: str = 'USR1@120'
    nodes: int = 1
    ntasks: int = 1
    mem: int = int(5e4)
    cpus_per_task: int = 1
    gpus: int = 1
    account: str = 'gpu-research'
    workspace: Path = PATHS.PROJECT_DIR
    outputs_relative_path: Path = PATHS.TENSORBOARD_DIR.relative_to(PATHS.PROJECT_DIR)
    master_port: str = DDP.MASTER_PORT
