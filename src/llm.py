import os
from dotenv import load_dotenv
from lyzr_automata.ai_models.model_base import AIModel
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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


def load_gemini_model(model_name="gemini-2.0-flash"):
    parameters = {
            "model": model_name,
        }

    gemini_model = GeminiModel(
        api_key=GEMINI_API_KEY,
        parameters=parameters
    )

    return gemini_model