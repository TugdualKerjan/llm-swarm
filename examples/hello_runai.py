import asyncio
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio


tasks = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the formula for water?"]
with LLMSwarm(
    LLMSwarmConfig(
        instances=1,
        inference_engine="tgi",
        job_scheduler="runai",
        gpus=1,
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        template_path="templates/tgi.template.yml",
        load_balancer_template_path="templates/nginx.template.conf",
        huggingface_token="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",
        model_max_total=3000,
        model_max_input=1200,
        per_instance_max_parallel_requests=300,
    )
) as llm_swarm:
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    print(client)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})

    async def process_text(task):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task},
            ],
            tokenize=False,
        )
        return await client.text_generation(
            prompt=prompt,
            max_new_tokens=200,
        )

    async def main():
        results = await tqdm_asyncio.gather(*(process_text(task) for task in tasks))
        df = pd.DataFrame({"Task": tasks, "Completion": results})
        print(df)

    asyncio.run(main())
