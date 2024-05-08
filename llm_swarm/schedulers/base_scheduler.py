from abc import ABC, abstractmethod
from llm_swarm.utils import LLMSwarmConfig
from typing import List, Optional, Tuple

class Scheduler(ABC):
    @abstractmethod
    def read_job_template(self, template_path: str) -> str:
        """
        Reads and returns the content of the job template file.

        Args:
            path (str): The path to the job template file.

        Returns:
            str: The content of the job template.
        """
        pass

    @abstractmethod
    def generate_job_config(self, config: LLMSwarmConfig, template: str) -> Tuple[str, str, str, str]:
        """Generates and returns job configurations based on the provided template.

        Args:
            config (LLMSwarmConfig): The LLMSwarmConfig object containing configuration parameters.
            template (str): The template string.

        Returns:
            Tuple[str, str, str, str]: A tuple containing the job timestamp, path, host_path and customized template.
        """
        pass
    
    @abstractmethod
    def start_jobs(self, path: str, template: str, job_timestamp: str, instances: int = 1) -> List[str]:
        """
        Starts Slurm jobs based on the provided job configuration.

        Args:
            config (LLMSwarmConfig): The LLMSwarmConfig object containing configuration parameters.
            path (str): The path to the job file.
            template (str): The customized template string.

        Returns:
            List[str]: A list of job IDs.
        """
        pass

    @abstractmethod
    def is_job_running(self, job_id: str) -> bool:
        """Checks if a job is still running."""
        pass

    @abstractmethod
    def make_sure_jobs_are_still_running(self, job_ids: List[str], log_path: str) -> None:
        """
        Checks if the specified job IDs are still running.

        Args:
            config (LLMSwarmConfig): The LLMSwarmConfig object containing configuration parameters.
            job_ids (List[str]): A list of job IDs to check.
        """
        pass

    @abstractmethod
    def get_endpoints(self, host_path: str, config: LLMSwarmConfig, instances: int = 1, job_ids: Optional[List[str]] = None) -> List[str]:
        """
        Retrieve the endpoints for the running jobs.

        Args:
            host_path (str): The path to the endpoint file.
            config (LLMSwarmConfig): The LLMSwarmConfig object containing configuration parameters.
            instances (int, optional): The number of instances. Defaults to 1.
            job_ids (Optional[List[str]], optional): A list of job IDs. Defaults to None.

        Returns:
            List[str]: A list of endpoints.
        """
        pass

    @abstractmethod
    def check_if_endpoint_reachable(self, endpoint: str) -> bool:
        """
        Check if the specified endpoint is reachable.

        Args:
            endpoint (str): The endpoint to check.

        Returns:
            bool: True if the endpoint is reachable, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup_jobs(self, job_ids: List[str]) -> None:
        """
        Cleanup jobs by cancelling them.

        Args:
            job_ids (List[str]): A list of job IDs to cancel.
        """
        pass
