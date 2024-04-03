import os
from haystack_integrations.components.generators.ollama import (
    OllamaChatGenerator
)

from .logger import get_logger


class AppConfig:
    def __init__(self):
        self.HUGGING_FACE_HUB_TOKEN = self._load_env_var(
            'HUGGING_FACE_HUB_TOKEN'
        )
        self.OLLAMA_MODEL_NAME = self._load_env_var('OLLAMA_MODEL_NAME')
        self.OLLAMA_URL = self._load_env_var('OLLAMA_URL')
        self.OLLAMA_TIMEOUT = self._load_env_var("OLLAMA_TIMEOUT", 120)

        self.OLLAMA_CHAT_URL = f"{self.OLLAMA_URL}/api/chat"

    def _load_env_var(self, name, default=None):
        """
        Loads and validates a single environment variable.

        Args:
            name (str): The name of the environment variable to load.

        KWArgs:
            default (str): Default value for environment variables.

        Returns:
            str: The value of the environment variable.

        Raises:
            ValueError: If the environment variable is missing.
        """
        value = os.environ.get(name, default)
        if value is None:
            raise ValueError(f'Missing environment variable: {name}')

        return value


# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)

config = AppConfig()
logger.debug(f'Using {config.OLLAMA_MODEL_NAME=}')
logger.debug(f'Endpoint: {config.OLLAMA_URL=}')
logger.debug(f'Generate: {config.OLLAMA_CHAT_URL=}')
logger.debug(f'Timeout: {config.OLLAMA_TIMEOUT=}')

logger.info(f"Setting up ollama with {config.OLLAMA_MODEL_NAME}")

llm = OllamaChatGenerator(
    model=config.OLLAMA_MODEL_NAME,
    url=config.OLLAMA_CHAT_URL,
    timeout=config.OLLAMA_TIMEOUT
)
