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

class HuggingFaceModel(AIModel):
    def __init__(self, api_key=None, parameters=None):
        # Note: HuggingFace doesn't always need api_key, but keeping for consistency
        self.api_key = api_key
        self.parameters = parameters or {}
        # attn_impl = "flash_attention_2" if supports_flash_attention() else "sdpa"
        # Load model with desired attention implementation here
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)  

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.parameters["model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation='eager',
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.parameters["model"],
            use_fast=True,
        )
        # Build the pipeline without passing attn_implementation
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=0,
        )

    def generate_text(self, task_id=None, system_persona=None, prompt=None):
        """
        Required abstract method implementation for text generation
        """
        # Combine system persona and prompt if both are provided
        full_prompt = prompt
        if system_persona and prompt:
            full_prompt = f"{system_persona}\n\n{prompt}"
        elif system_persona:
            full_prompt = system_persona
            
        outputs = self.pipeline(
            full_prompt,
            max_new_tokens=self.parameters.get("max_tokens", 1500),
            do_sample=True,
            temperature=self.parameters.get("temperature", 0.6),
            top_p=self.parameters.get("top_p", 0.9),
        )
        
        return outputs[0]['generated_text']

    def generate_image(self, task_id=None, prompt=None, resource_box=None, tasks=None):
        """
        Required abstract method implementation for image generation
        Since HuggingFace text models don't generate images, raise an appropriate error
        """
        raise NotImplementedError(
            "Image generation is not supported by this HuggingFace text model. "
            "Use a different model or service for image generation tasks."
        )

    def log_and_get_completion(self, prompt: str) -> str:
        """
        Your custom method - this can remain as additional functionality
        """
        return self.generate_text(prompt=prompt)

# --- Initialize Models and Clients ---
huggingface_model = HuggingFaceModel(
    # api_key=HUGGINGFACE_API_KEY,
    parameters={
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

qdrant_client = QdrantClient(host="localhost", port=6333)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- Define Tools ---
def search_knowledge_base(query: str):
    """Searches the customer support knowledge base for relevant information."""
    query_vector = list(embedding_model.embed([query]))[0]
    search_result = qdrant_client.search(
        collection_name="customer_support_kb",
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )
    context = "\n".join([hit.payload['utterance'] for hit in search_result])
    return context

# In-memory conversation storage (replace with a DB or Qdrant in prod)
conversation_history = {}

def save_message(session_id: str, role: str, content: str):
    """Saves a message to the conversation history for a given session."""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"role": role, "content": content})
    return "Message saved."

def get_history(session_id: str):
    """Retrieves the conversation history for a given session."""
    return conversation_history.get(session_id, [])

# --- Define Agents ---
manager_agent = Agent(
    role="Customer Support Manager",
    prompt_persona="You are an expert customer support manager. Your job is to understand the user's query, use the conversation history for context, and then use the knowledge base to find the most relevant answer. Finally, formulate a helpful and concise response."
)

sentiment_agent = Agent(
    role="Sentiment Analysis Expert",
    prompt_persona="You are an expert in analyzing text to determine its emotional tone. Classify the user's sentiment as Positive, Negative, or Neutral. Respond with only one of these words."
)

# --- Define Task ---
def create_support_task(user_query, session_id):
    history = get_history(session_id)
    context = search_knowledge_base(user_query)
    
    save_message(session_id, "user", user_query)

    instructions = f"""
    You are a customer support agent. A user has asked the following question: '{user_query}'
    
    Here is the conversation history for context:
    {history}
    
    Here is some relevant information from our knowledge base:
    {context}
    
    Based on all of this information, please provide a clear and helpful response to the user's question.
    """

    support_task = Task(
        name="Customer Support Task",
        agent=manager_agent,
        model=huggingface_model,
        instructions=instructions,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT,
    )
    return support_task
