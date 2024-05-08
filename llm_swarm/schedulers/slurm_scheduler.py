from .base_scheduler import Scheduler
from llm_swarm.utils import run_command, Loader, LLMSwarmConfig
import time
import os
from typing import List, Optional, Tuple
from time import sleep
from huggingface_hub import get_session


class SlurmScheduler(Scheduler):
    def read_job_template(self, template_path: str) -> str:
        with open(template_path) as f:
            return f.read()
    
    def generate_job_config(self, config: LLMSwarmConfig, template: str) -> Tuple[str, str, str, str]:
        job_timestamp = f"{int(time.time())}"
        path = os.path.join(config.logs_folder, f"{job_timestamp}_{config.inference_engine}.slurm")
        host_path = os.path.join(config.logs_folder, f"{job_timestamp}_host_{config.inference_engine}.txt")
         # Customize the template
        template = template.replace(r"{{HUGGING_FACE_HUB_TOKEN}}", config.huggingface_token or "")
        template = template.replace(r"{{hosts_path}}", host_path)
        template = template.replace(r"{{model}}", config.model)
        template = template.replace(r"{{port}}", str(config.port))
        template = template.replace(r"{{gpus}}", str(config.gpus))
        template = template.replace(r"{{model_max_output}}", str(config.model_max_total))
        template = template.replace(r"{{model_input_length}}", str(config.model_max_input))
        
        return job_timestamp, path, host_path, template
    
    def start_jobs(self, path: str, template: str, job_timestamp: str, instances: int = 1) -> List[str]:
        with open(path, "w") as f:
            f.write(template)
        return [run_command(f"sbatch --parsable {path}") for _ in range(instances)]
         

    def is_job_running(self, job_id: str) -> bool:
        command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
        try:
            my_running_jobs = run_command(command).splitlines()
            return job_id in my_running_jobs
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            return False
    

    def make_sure_jobs_are_still_running(self, job_ids: List[str], log_path: str) -> None:
        if job_ids:
            for job_id in job_ids:
                if not self.is_job_running(job_id):
                    slurm_log_path = os.path.join(log_path, f"llm-swarm_{job_id}.out")
                    print(f"\n❌ Failed! Job {job_id} is not running; checkout {slurm_log_path} ")
                    raise RuntimeError(f"Job {job_id} is not running")


    def get_endpoints(self, host_path: str, config: LLMSwarmConfig, instances: int = 1, job_ids: Optional[List[str]] = None) -> List[str]:
        trying = True
        with Loader(f"Waiting for {host_path} to be created"):
            while trying:
                try:
                    endpoints = open(host_path).read().splitlines()
                    assert len(endpoints) == instances, f"#endpoints {len(endpoints)} doesn't match #instances {instances}"
                    # due to race condition (slurm writing & us reading)
                    trying = False
                except (OSError, AssertionError):
                    self.make_sure_jobs_are_still_running(job_ids, config)
                    sleep(1)

    def check_if_endpoint_reachable(self, endpoint: str) -> bool:
        get_session().get(f"{endpoint}/health") #TODO: Might not be the same for runai
        print(f"\nConnected to {endpoint}")
        return True

    def cleanup_jobs(self, job_ids: List[str]):
        for job_id in job_ids:
            run_command(f"scancel {job_id}")
        pass
