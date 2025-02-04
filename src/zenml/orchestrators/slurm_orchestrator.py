from typing import Any, Dict, Optional, Type, cast
from uuid import uuid4

from src.consts import PATHS
from src.utils.slurm import submit_job
from zenml.config.base_settings import BaseSettings
from zenml.logger import get_logger
from zenml.models import PipelineDeploymentResponse
from zenml.orchestrators.base_orchestrator import (
    BaseOrchestratorFlavor,
)
from zenml.orchestrators.local.local_orchestrator import LocalOrchestrator
from zenml.stack import Stack

logger = get_logger(__name__)


class SlurmOrchestratorSettings(BaseSettings):
    """Slurm orchestrator settings."""

    job_name: str = "zenml_slurm_job"
    gpu_type: str = "titan_xp-studentrun"
    timeout_min: int = 1200
    memory_required: Optional[str] = None
    slurm_nodes: int = 1
    tasks_per_node: int = 1
    slurm_cpus_per_task: int = 1
    slurm_gpus_per_node: int = 1
    slurm_nodelist: Optional[str] = None


class SlurmOrchestrator(LocalOrchestrator):
    """Orchestrator for running pipelines on Slurm using submitit."""

    @property
    def settings_class(self) -> Optional[Type[SlurmOrchestratorSettings]]:
        """Settings class for the Local Docker orchestrator.

        Returns:
            The settings class.
        """
        return SlurmOrchestratorSettings

    def prepare_or_run_pipeline(
        self,
        deployment: PipelineDeploymentResponse,
        stack: Stack,
        environment: Dict[str, str],
    ) -> Any:
        """Submit the pipeline as a single Slurm job.

        Args:
            deployment: The pipeline deployment to prepare or run.
            stack: The stack on which the pipeline is deployed.
            environment: Environment variables to set.
        """
        self._orchestrator_run_id = str(uuid4())

        settings = cast(SlurmOrchestratorSettings, self.get_settings(deployment))

        def run_pipeline():
            """Function to run the pipeline steps."""
            return super(SlurmOrchestrator, self).prepare_or_run_pipeline(deployment, stack, environment)

        # Submit the job to Slurm
        job = submit_job(
            func=run_pipeline,
            gpu_type=settings.gpu_type,
            job_name=settings.job_name,
            log_folder=str(PATHS.SLURM_DIR / settings.job_name / "%j"),
            timeout_min=settings.timeout_min,
            memory_required=settings.memory_required,
            slurm_nodes=settings.slurm_nodes,
            tasks_per_node=settings.tasks_per_node,
            slurm_cpus_per_task=settings.slurm_cpus_per_task,
            slurm_gpus_per_node=settings.slurm_gpus_per_node,
            slurm_nodelist=settings.slurm_nodelist,
        )

        logger.info(f"Submitted Slurm job with ID: {job.job_id} and name: {settings.job_name}")
        return job.result()


class SlurmOrchestratorFlavor(BaseOrchestratorFlavor):
    """Flavor for the Slurm orchestrator."""

    @property
    def name(self) -> str:
        """The flavor name.

        Returns:
            The flavor name.
        """
        return "slurm"

    @property
    def implementation_class(self) -> Type[SlurmOrchestrator]:
        """Implementation class for this flavor.

        Returns:
            The implementation class.
        """
        return SlurmOrchestrator
