from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import NewType
from typing import Optional
from typing import TypedDict

from src.utils.types_utils import STREnum


class SPLIT(STREnum):
    DEV = 'dev'
    TRAIN = 'train'
    TEST = 'test'


class T_SAMPLER(STREnum):
    UNIFORM = 'uniform'
    CONSTANT = 'constant'


class SAMPLERS(STREnum):
    STANDARD = 'standard'
    DPM_SOLVER_PP = 'DPMSolver++'
    FAST_DPM = 'FastDPM'
    DDIM = 'DDIM'


class MODEL(STREnum):
    DDPM = 'ddpm'
    EDM = 'edm'


class CONFIG_KEYS(STREnum):
    SAMPLERS = 'samplers'
    TRAINING = 'training'
    DDPM = MODEL.DDPM.value
    EDM = MODEL.EDM.value
    FASHION_MNIST = 'fashion_mnist'


class STEP_TIMING(STREnum):
    BATCH = 'batch'
    EPOCH = 'epoch'
    EVALUATION = 'evaluation'


class LR_SCHEDULER(STREnum):
    STEP = 'step'
    ONE_CYCLE = 'one_cycle'


class OPTIMIZER(STREnum):
    ADAM = 'adam'
    ADAMW = 'adamw'


class METRICS(STREnum):
    LOSS = 'loss'


TimeStep = NewType('TimeStep', int)
IEarlyStopped = NewType('IEarlyStopped', bool)
ILoss = NewType('ILoss', float)
IMetrics = NewType('IMetrics', Dict[str, float])


class ITrainArgs(NamedTuple):
    experiment_name: str
    config: "Config"  # TODO: to avoid circular, fix it
    run_id: Optional[str]


class Checkpoint(TypedDict):
    epoch: int
    total_steps: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    best_loss: float
