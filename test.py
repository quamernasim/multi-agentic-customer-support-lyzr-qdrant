# agents.py
import os
import torch
import transformers
from dotenv import load_dotenv
from lyzr_automata.ai_models.model_base import AIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from lyzr_automata.tasks.task_literals import InputType, OutputType
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

# Load environment variables
load_dotenv()
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- Define Custom HuggingFace Model Class ---
import transformers
import torch

# def supports_flash_attention(device: int = 0) -> bool:
#     major, minor = torch.cuda.get_device_capability(device)
#     # Ampere: SM 8.0+ or SM 9.0 (Hopper)
#     return (major == 8 and minor >= 0) or (major >= 9)

from transformers import BitsAndBytesConfig  


# parameters={
#     "model": "meta-llama/Llama-3.2-3B-Instruct",
#     "temperature": 0.2,
#     "max_tokens": 1500,
# }
# # attn_impl = "flash_attention_2" if supports_flash_attention() else "sdpa"
# # Load model with desired attention implementation here
# bnb_config = BitsAndBytesConfig(load_in_4bit=True)  

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     parameters["model"],
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     attn_implementation='eager',
#     # quantization_config=bnb_config,
#     # low_cpu_mem_usage=True,
# )
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     parameters["model"],
#     use_fast=False,
# )
# # Build the pipeline without passing attn_implementation
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     # device=0,
# )

import google.generativeai as genai

# Load API key
genai.configure(api_key="")

def generate_text():
    """
    Required abstract method implementation for text generation
    """
    # Combine system persona and prompt if both are provided
    full_prompt = f"""
In your role as Sentiment Analysis Expert, you embody a persona defined by You are a sentiment classifier. Your ONLY task is to classify text.
    You MUST respond with ONLY one of the following words: Positive, Negative, or Neutral.
    Do not provide any explanation or extra text..

Now execute these instructions: Classify the sentiment of this user query: i want to return my mobile Input: None
"""
    print(f'[DEBUG - IN] - {full_prompt}')
    # outputs = pipeline(
    #     full_prompt,
    #     max_new_tokens=parameters.get("max_tokens", 1500),
    #     do_sample=False,
    #     temperature=parameters.get("temperature", 0.2),
    #     top_p=parameters.get("top_p", 0.9),
    # )
    # out = outputs[0]['generated_text']
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        # system_instruction="You are a helpful assistant for geophysics researchers.",
    )

    # chat = model.start_chat()

    out = model.generate_content(full_prompt)
    print(f'[DEBUG - OUT] - {out.text}')
    # return outputs[0]['generated_text']

generate_text()























