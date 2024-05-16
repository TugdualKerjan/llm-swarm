from llm_swarm import LLMSwarmConfig, LLMSwarm
from transformers import HfArgumentParser

parser = HfArgumentParser(LLMSwarmConfig)
config = parser.parse_args_into_dataclasses()[0]
with LLMSwarm(config) as llm_swarm:
    try:
        while True:
            input("Press Enter to EXIT...")
            break
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Exiting...")
    finally:
        print("LLMSwarm stopped")