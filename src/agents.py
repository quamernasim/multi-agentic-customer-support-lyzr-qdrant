# agents.py
import os
from dotenv import load_dotenv
from lyzr_automata.ai_models.model_base import AIModel
from lyzr_automata import Agent
import google.generativeai as genai
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Load environment variables
load_dotenv()

# Load API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Define parameters for the Gemini model
parameters = {
        "model": "gemini-2.0-flash",
    }

# --- Define Custom Gemini Model Class ---
class GeminiModel(AIModel):
    def __init__(self, api_key=None, parameters=None):
        self.api_key = api_key
        self.parameters = parameters or {}

        self.model = genai.GenerativeModel(
        model_name=self.parameters.get("model", "gemini-1.5-flash"),
        # system_instruction="You are a helpful assistant for geophysics researchers.",
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
    
        outputs = self.model.generate_content(full_prompt).text
        print(f"Generated text: {outputs}")
        return outputs

    def generate_image(self, task_id=None, prompt=None, resource_box=None, tasks=None):
        """
        Required abstract method implementation for image generation
        """
        raise NotImplementedError(
            "Image generation is not supported by this model."
        )

    def log_and_get_completion(self, prompt: str) -> str:
        """
        Your custom method - this can remain as additional functionality
        """
        return self.generate_text(prompt=prompt)

# --- Initialize Models and Clients ---
gemini_model = GeminiModel(
    api_key=GEMINI_API_KEY,
    parameters=parameters
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
# --- Define Agents with Improved Prompts ---
manager_agent = Agent(
    role="Customer Support Manager",
    prompt_persona="You are a helpful Customer Support AI assistant."
)

sentiment_agent = Agent(
    role="Sentiment Analysis Expert",
    prompt_persona="""You are a sentiment classifier. Your ONLY task is to classify text.
    You MUST respond with ONLY one of the following words: Positive, Negative, or Neutral.
    Do not provide any explanation or extra text."""
)
