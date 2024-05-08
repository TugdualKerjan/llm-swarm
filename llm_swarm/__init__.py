import os
from .utils import run_command, get_unused_port, Loader, test_generation, LLMSwarmConfig
from typing import List, Optional
import requests
from time import sleep
from .schedulers.slurm_scheduler import SlurmScheduler
from .schedulers.runai_scheduler import RunaiScheduler
from huggingface_hub import get_session

class LLMSwarm:
    def __init__(self, config: LLMSwarmConfig) -> None:
        self.config = config
        self.scheduler = self._create_scheduler()
        self.cleaned_up = False
        self.endpoint = None  # Initialize to None
        self._create_logs_folder()
        self._handle_debug_endpoint()

    def _create_logs_folder(self):
        if not os.path.exists(self.config.logs_folder):
            os.makedirs(self.config.logs_folder)

    def _create_scheduler(self):
        return SlurmScheduler() if self.config.job_scheduler == "slurm" else RunaiScheduler()

    def _handle_debug_endpoint(self):
        """Assign config attributes to local variables for clarity"""
        debug_endpoint = self.config.debug_endpoint
        inference_engine = self.config.inference_engine
        instances = self.config.instances
        per_instance_max_parallel_requests = self.config.per_instance_max_parallel_requests
        
        if debug_endpoint:
            # Use debug endpoint as is
            self.endpoint = debug_endpoint
            if inference_engine == "vllm":
                self.endpoint = f"{debug_endpoint}/generate"
            
            # Set suggested max parallel requests based on debug endpoint
            if debug_endpoint.startswith("https://api-inference.huggingface.co/"):
                self.suggested_max_parallel_requests = 40
            else:
                self.suggested_max_parallel_requests = per_instance_max_parallel_requests * instances
        else:
            # Default behavior when debug endpoint is not provided
            self.suggested_max_parallel_requests = per_instance_max_parallel_requests * instances


    def start(self):
        # template = self.scheduler.read_job_template(self.config.template_path)

        # job_timestamp, path, host_path, template = self.scheduler.generate_job_config(self.config, template)

        # job_ids = self.scheduler.start_jobs(path, template, job_timestamp, self.config.instances)
        
        job_ids = ["runai-1715173527-2", "runai-1715173527-1", "runai-1715173527-2"]
        host_path = ""
        self._wait_for_jobs_to_start(job_ids)
        self._retrieve_endpoints_and_test(host_path, job_ids)

        # print(f"{self.config.job_scheduler} Job ID: {self.job_ids}")
        # print(f"📖 {self.config.job_scheduler} hosts path: {host_path}")

        if len(self.endpoints) == 1:
            self.endpoint = self.endpoints[0]
        else:
            self._run_load_balancer()

        print(f"🔥 endpoint ready {self.endpoint}")

        if self.config.inference_engine == "vllm":
            self.endpoint = f"{self.endpoint}/generate"

    def _wait_for_jobs_to_start(self, job_ids: List[str]):
        for job_id in job_ids:
            with Loader(f"Waiting for {job_id} to be created"):
                while not self.scheduler.is_job_running(job_id):
                    sleep(1)
            log_path = os.path.join(self.config.logs_folder, f"llm-swarm_{job_id}.out")
            print(f"📖 {self.config.job_scheduler} log path: {log_path}")

    def _retrieve_endpoints_and_test(self, host_path: str, job_ids: List[str]) -> None:
        self.endpoints = self.scheduler.get_endpoints(host_path, self.config, self.config.instances, job_ids)
        print(f"Endpoints running properly: {self.endpoints}")
        for endpoint in self.endpoints:
            test_generation(endpoint)

    def _run_loadbalancer(self):
        # run a load balancer
        with open(self.config.load_balancer_template_path) as f:
            # templates/nginx.template.conf
            load_balancer_template = f.read()
        servers = "\n".join([f"server {endpoint.replace('http://', '')};" for endpoint in self.endpoints])
        unused_port = get_unused_port()
        load_balancer_template = load_balancer_template.replace(r"{{servers}}", servers)
        load_balancer_template = load_balancer_template.replace(r"{{port}}", str(unused_port))
        load_balancer_path = os.path.join(self.config.job_scheduler, f"{self.filename}_load_balancer.conf")
        with open(load_balancer_path, "w") as f:
            f.write(load_balancer_template)
        load_balance_endpoint = f"http://localhost:{unused_port}"
        command = f"docker run -d -p {unused_port}:{unused_port} --network host -v $(pwd)/{load_balancer_path}:/etc/nginx/nginx.conf nginx"
        load_balance_endpoint_connected = False
        # run docker streaming output while we validate the endpoints
        self.container_id = run_command(command)
        last_line = 0
        while True:
            logs = run_command(f"docker logs {self.container_id}")
            lines = logs.split("\n")
            for line in lines[last_line:]:
                print(line)
            last_line = len(lines)

            if not load_balance_endpoint_connected:
                try:
                    get_session().get(f"{load_balance_endpoint}/health")
                    print(f"🔥 endpoint ready {load_balance_endpoint}")
                    load_balance_endpoint_connected = True
                    self.endpoint = load_balance_endpoint
                    break
                except requests.exceptions.ConnectionError:
                    sleep(1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()

    def get_endpoints(self, endpoint_path: str, config: LLMSwarmConfig, instances: int = 1, job_ids: Optional[List[str]] = None) -> List[str]:
        """Return list of endpoints from either a file or a comma separated string.
        It also checks if the endpoints are reachable.

        Args:
            endpoint_path (str): path to file containing endpoints or comma separated string
            instances (int, optional): number of instances. Defaults to 1.

        Returns:
            List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
        """
        endpoints = self.scheduler.get_endpoints(endpoint_path, config, instances, job_ids)
        for endpoint in endpoints:
            connected = False
            while not connected:
                try:
                    self.scheduler.check_if_endpoints_reachable(endpoint)
                except requests.exceptions.ConnectionError:
                    self.scheduler.make_sure_jobs_are_still_running(self, job_ids, config)
                    sleep(1)

    def cleanup(self):
        if self.config.debug_endpoint:
            return
        if self.cleaned_up:
            return
        else:
            self.scheduler.cleanup_jobs(self.job_ids)

        print("inference instances terminated")
        
        if self.container_id:
            run_command(f"docker kill {self.container_id}")
            print("docker process terminated")

        self.cleaned_up = True


if __name__ == "__main__":
    with LLMSwarm(
        LLMSwarmConfig(
            instances=3,
            inference_engine="tgi",
            job_scheduler="slurm",
            slurm_template_path="templates/tgi_h100.template.slurm",
            load_balancer_template_path="templates/nginx.template.conf",
        )
    ) as llm_swarm:
        while True:
            input("Press Enter to EXIT...")
            break
