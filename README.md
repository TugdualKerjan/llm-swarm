<div align="center">
  <h1>🐝 Fork of llm-swarm 🦋</h1>
  <p><em>Manage scalable open LLM inference endpoints in <b>Runai</b> and Slurm clusters</em></p>
</div>

## Features

- **😎 This fork implements the management for RunAI clusters as well. 😎**
- Generate synthetic datasets for pretraining or fine-tuning using either local LLMs or [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated) on the Hugging Face Hub.
- Integrations with [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) and [vLLM](https://github.com/vllm-project/vllm) to generate text at scale.

## What's different here ?

* Support for RunAI Schedulers is added.

* The code is agnostic to a specific scheduler, new ones can be added following the [BaseScheduler.py](llm_swarm/schedulers/base_scheduler.py) class

* Templates have been cleaned up and an example for running with RunAI is given.

* [\_\_init__.py](llm_swarm/__init__.py) is more readable. 

* [utils.py](llm_swarm/utils.py) is a file full of helper functions

* Typing is used to avoid type errors in functions

## Install and prepare

```bash
pip install -e .
# or pip install llm_swarm
mkdir -p .cache/
# you can customize the above docker image cache locations and change them in `templates/tgi_h100.template.slurm` and `templates/vllm_h100.template.slurm`
```

## For the rest read the [official README.md](https://github.com/huggingface/llm-swarm/tree/main)