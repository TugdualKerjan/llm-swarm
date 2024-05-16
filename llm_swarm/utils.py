import socket
import requests
import subprocess
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from typing import Literal, Optional, TypeVar
from dataclasses import dataclass, field


def run_command(command: str):
    # print(f"running {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, errors = process.communicate()
    return_code = process.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"
    return output.decode("utf-8").strip()

def get_unused_port(start=50000, end=65535):
    for port in range(start, end + 1):
        try:
            sock = socket.socket()
            sock.bind(("", port))
            sock.listen(1)
            sock.close()
            return port
        except OSError:
            continue
    raise IOError("No free ports available in range {}-{}".format(start, end))

class Loader:
    def __init__(self, desc="Loading...", end="👌 Done!", failed="👎 Aborted!", timeout=0.2):
        """
        A loader-like context manager
        Modified from https://stackoverflow.com/a/66558182/6611317

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            failed (str, optional): Final print on failure. Defaults to "Aborted!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end + " " + self.desc
        self.failed = failed + " " + self.desc
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        # self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.steps = ["👉  👈", " 👉👈 "]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        try:
            for c in cycle(self.steps):
                if self.done:
                    break
                print(f"\r{c} {self.desc}", flush=True, end="")
                sleep(self.timeout)
        except KeyboardInterrupt:
            self.stop()
            print("KeyboardInterrupt by user")

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.stop()
        else:
            self.done = True
            cols = get_terminal_size((80, 20)).columns
            print("\r" + " " * cols, end="", flush=True)
            print(f"\r{self.failed}", flush=True)

DataclassT = TypeVar("DataclassT")

@dataclass
class LLMSwarmConfig:
    instances: int = 1
    inference_engine: Literal["tgi", "vllm"] = "tgi"
    job_scheduler: Literal["slurm", "runai"] = "slurm"
    template_path: Optional[str] = "templates/tgi_h100.template.slurm"
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    revision: str = "main"
    gpus: float = 0.4
    load_balancer_template_path: Optional[str] = "templates/nginx.template.conf"
    per_instance_max_parallel_requests: int = 128
    debug_endpoint: Optional[str] = None
    huggingface_token: Optional[str] = None
    model_max_input: int = 200
    model_max_total: int = 300
    port: int = 6969
    logs_folder: str = "logs"

    def __post_init__(self):
        if not (1024 <= self.port <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        if self.gpus <= 0:
            raise ValueError("Number of GPUs must be greater than zero")