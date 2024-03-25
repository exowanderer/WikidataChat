import os
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from .logger import get_logger

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)

HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN')
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"

logger.info(f'Using {OLLAMA_MODEL_NAME=}')
logger.info(f'Endpoint: {OLLAMA_URL=}')
logger.info(f'Generate: {OLLAMA_CHAT_URL=}')

logger.info(f"Setting up ollama with {OLLAMA_MODEL_NAME}")

llm = OllamaChatGenerator(
    model=OLLAMA_MODEL_NAME,
    url=OLLAMA_CHAT_URL,
    timeout=120
)
