
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from langchain.llms import HuggingFacePipeline

MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
print(generation_config)
print('----------------', model.device)

generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    generation_config = generation_config,
)

llm = HuggingFacePipeline(pipeline = generation_pipeline)


from langchain.chains.summarize import load_summarize_chain
from prompts.summary_prompts import map_template, reduse_template

summarize_chain = load_summarize_chain(llm = llm, chain_type="map_reduce", map_prompt = map_template, combine_prompt = reduse_template, verbose=True)


