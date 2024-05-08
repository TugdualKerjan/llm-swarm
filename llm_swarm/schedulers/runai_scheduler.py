from .base_scheduler import Scheduler
from llm_swarm.utils import run_command, Loader, LLMSwarmConfig
import time
import os
from typing import List, Optional, Tuple
from time import sleep
import requests

class RunaiScheduler(Scheduler):
    def read_job_template(self, template_path: str) -> str:
        with open(template_path) as f:
            return f.read()

    def generate_job_config(self, config: LLMSwarmConfig, template: str) -> Tuple[str, str, str, str]:
        job_timestamp = f"{int(time.time())}"
        path = os.path.join(config.logs_folder, f"{job_timestamp}_{config.inference_engine}.yml")
        host_path = os.path.join(config.logs_folder, f"{job_timestamp}_host_{config.inference_engine}.txt")

        # Customize the template
        template = template.replace(r"{{HUGGING_FACE_HUB_TOKEN}}", config.huggingface_token or "")
        template = template.replace(r"{{hosts_path}}", host_path)
        template = template.replace(r"{{model}}", config.model)
        template = template.replace(r"{{port}}", str(config.port))
        template = template.replace(r"{{gpus}}", str(config.gpus))
        template = template.replace(r"{{model_max_output}}", str(config.model_max_total))
        template = template.replace(r"{{model_input_length}}", str(config.model_max_input))

        return job_timestamp, path, "", template


    def start_jobs(self, path: str, template: str, job_timestamp: str, instances: int = 1) -> List[str]:
        job_ids = []
        for i in range(instances):
            job_name = f"runai-{job_timestamp}-{i}"
            with open(path, "w") as f:
                f.write(template.replace(r"{{job_name}}", job_name))
            run_command(f"kubectl create -f {path}")
            job_ids.append(job_name)
        return job_ids

    def is_job_running(self, job_id: str) -> bool:
        # Run the command to list jobs
        command = "runai list jobs -A"
        try:
            # Execute the command and capture the output
            result = run_command(command)
            # result = subprocess.run(command, shell=True, text=True, capture_output=True)
            
            # # Check for command execution errors
            # if result.returncode != 0:
            #     print(f"Error running command: {result.stderr.strip()}")
            #     return False

            # Parse the command's output
            # lines = result.stdout.splitlines()
            lines = result.splitlines()
            for line in lines:
                # Skip the header line
                if line.startswith("NAME"):
                    continue
                
                # Extract job details
                columns = line.split()
                name = columns[0]
                status = columns[1]
                
                # Check if the job name matches and the status is Running
                if name == job_id and status == "Running":
                    return True
            
            return False

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            return False

    def make_sure_jobs_are_still_running(self, job_ids: List[str], log_path: str) -> None:
        if job_ids:
            for job_id in job_ids:
                if self.is_job_running(self, job_id):
                    print(f"\n❌ Failed! Job {job_id} is not running; Checkout the logs with $runai logs {job_id}")
                    raise RuntimeError(f"Job {job_id} is not running")

    def get_endpoints(self, host_path: str, config: LLMSwarmConfig, instances: int = 1, job_ids: Optional[List[str]] = None) -> List[str]:
        trying = True
        with Loader(f"Waiting for {host_path} to be created"):
            while trying:
                try:
                    endpoints = []
                    # Run the command to list jobs
                    command = "runai list jobs -A"
                    # Execute the command and capture the output
                    result = run_command(command)
                    
                    # Parse the command's output
                    lines = result.splitlines()
                    for line in lines:
                        # Skip the header line
                        if line.startswith("NAME"):
                            continue
                        
                        # Extract job details
                        columns = line.split()
                        name = columns[0]
                        endpoint = columns[3]
                        
                        # Check if the job name matches and the status is Running
                        if name in job_ids and endpoint != "-":
                            endpoints.append(f"http://{endpoint}:{config.port}")

                    assert (
                        len(endpoints) == instances
                    ), f"#endpoints {len(endpoints)} doesn't match #instances {instances}"  # could read an empty file
                    # due to race condition (slurm writing & us reading)
                    trying = False
                except (OSError, AssertionError) as e:
                    print(e)
                    self.make_sure_jobs_are_still_running(job_ids, config)
                    sleep(1)
        pass

    def check_if_endpoint_reachable(self, endpoint: str) -> bool:
        try:
            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "inputs": "What is Deep Learning?",
                "parameters": {
                    "max_new_tokens": 200,
                },
            }
            requests.post(endpoint, headers, json=data)
            print(f"\nConnected to {endpoint}")
            return True
        except requests.exceptions.ConnectionError:
            return False

    def cleanup_jobs(self, job_ids: List[str]):
        for job_id in job_ids:
            run_command(f"runai delete job {job_id}")
        pass
