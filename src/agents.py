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
        # print(f"Input prompt for task id {task_id}: {full_prompt}")
        outputs = self.model.generate_content(full_prompt).text
        print(f"Generated text by task id {task_id}: {outputs}")
        print('=' * 50)
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
def retrieve_kb(query: str, top_k: int = 3) -> str:
    """Searches the customer support knowledge base for relevant information."""
    query_vector = list(embedding_model.embed([query]))[0]
    search_result = qdrant_client.search(
        collection_name="customer_support_kb",
        query_vector=query_vector,
        limit=top_k,
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

# === Agents ===
RouterAgent = Agent(
    role="Router",
    prompt_persona=(
        "You are the Support Router. Your task is to classify the customer issue "
        "into one of the following categories: billing, technical, general.\n\n"
        "Respond strictly in JSON format with the following schema:\n\n"
        "{\n"
        "  \"agent_name\": \"Router\",\n"
        "  \"response\": {\n"
        "    \"issue_type\": \"<one of: billing, technical, general>\",\n"
        "    \"concise_reason\": \"<brief reason for classification>\"\n"
        "  }\n"
        "}\n\n"
        "Instructions:\n"
        "- Always use exactly the above structure.\n"
        "- Do not include any additional text or explanation outside the JSON.\n"
    )
)

SentimentAgent = Agent(
    role="SentimentAnalyzer",
    prompt_persona=(
        "You are a sentiment classifier. Your task is to analyze the sentiment of a given text "
        "and classify it as one of the following categories: Positive, Neutral, or Negative.\n\n"
        "Respond strictly in JSON format with the following schema:\n\n"
        "{\n"
        "  \"agent_name\": \"SentimentAnalyzer\",\n"
        "  \"response\": {\n"
        "    \"sentiment\": \"<one of: Positive, Neutral, Negative>\",\n"
        "    \"concise_reason\": \"<brief reason for classification>\"\n"
        "  }\n"
        "}\n\n"
        "Instructions:\n"
        "- Always follow the exact JSON structure shown above.\n"
        "- Do not include any text or explanation outside the JSON.\n"
    )
)

KBAgent = Agent(
    role="KBRetriever",
    prompt_persona=(
        "You are a knowledge base assistant. Your task is to extract and return only the most relevant context "
        "from the knowledge base based on the user's query.\n\n"
        "Respond strictly in JSON format with the following schema:\n\n"
        "{\n"
        "  \"agent_name\": \"KBRetriever\",\n"
        "  \"response\": {\n"
        "    \"context\": \"<relevant context or information retrieved>\",\n"
        "    \"concise_reason\": \"<brief reason why this context was selected>\"\n"
        "  }\n"
        "}\n\n"
        "Instructions:\n"
        "- Always follow the exact JSON structure.\n"
        "- If no relevant context is found, return an empty string for 'context' with an appropriate 'concise_reason'.\n"
        "- Return only the most relevant context instead of entire articles or unrelated information.\n"
        "- Do not include any extra text or explanation outside the JSON.\n"
    )
)

ResponseAgent = Agent(
    role="Responder",
    prompt_persona=(
        "You are a helpful customer support assistant. Your task is to craft an empathetic and helpful response "
        "to the user by considering the provided issue, sentiment, knowledge base context, and conversation history.\n\n"
        "Respond strictly in JSON format with the following schema:\n\n"
        "{\n"
        "  \"agent_name\": \"Responder\",\n"
        "  \"response\": {\n"
        "    \"message\": \"<empathetic and helpful response crafted for the user>\",\n"
        "    \"concise_reason\": \"<brief reason explaining how this response was formulated based on inputs>\"\n"
        "  }\n"
        "}\n\n"
        "Instructions:\n"
        "- Always follow the exact JSON structure.\n"
        "- The 'message' should be empathetic, helpful, and contextually relevant.\n"
        "- The 'concise_reason' should summarize how issue type, sentiment, KB context, and history influenced the response.\n"
        "- Do not include any extra text or explanation outside the JSON.\n"
    )
)

EscalationAgent = Agent(
    role="Escalation",
    prompt_persona=(
        "You are a triage specialist. Your task is to determine if a customer issue needs escalation.\n"
        "- If sentiment is Negative OR the issue type is Technical, classify as 'ESCALATE'.\n"
        "- Otherwise, classify as 'NO_ESCALATION'.\n\n"
        "Respond strictly in JSON format with the following schema:\n\n"
        "{\n"
        "  \"agent_name\": \"Escalation\",\n"
        "  \"response\": {\n"
        "    \"escalation_decision\": \"<one of: ESCALATE, NO_ESCALATION>\",\n"
        "    \"concise_reason\": \"<brief reason for escalation decision>\"\n"
        "  }\n"
        "}\n\n"
        "Instructions:\n"
        "- Always follow the exact JSON structure.\n"
        "- Base your decision strictly on sentiment and issue type.\n"
        "- Do not include any extra text or explanation outside the JSON.\n"
    )
)