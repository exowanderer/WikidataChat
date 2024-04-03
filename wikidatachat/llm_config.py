import os  # os module interacts with the OS environment variables.

from haystack_integrations.components.generators.ollama import (
    # Import the OllamaChatGenerator class for chat generation.
    OllamaChatGenerator
)

# Import the get_logger function from the local logger module.
from .logger import get_logger


class EnvConfig:
    def __init__(self):
        """
        Initializes the EnvConfig class, loading and validating required configurations
        from environment variables.
        """

        # Load and validate environment variables for application configuration.
        self.HUGGING_FACE_HUB_TOKEN = self._load_env_var(
            'HUGGING_FACE_HUB_TOKEN'
        )
        self.OLLAMA_MODEL_NAME = self._load_env_var('OLLAMA_MODEL_NAME')
        self.OLLAMA_URL = self._load_env_var('OLLAMA_URL')

        # Load and validate environment variable for timeout,
        #   with a default of 120 if not set.
        self.OLLAMA_TIMEOUT = self._load_env_var(
            name="OLLAMA_TIMEOUT",
            default=120,
            typecast=int
        )

        # Construct the Ollama chat URL from the base URL.
        self.OLLAMA_CHAT_URL = f"{self.OLLAMA_URL}/api/chat"

    def _load_env_var(self, name, default=None, typecast=None):
        """
        Loads an environment variable, with an option to provide 
            a default value.
        If no default is provided and the variable is missing, raises an error.

        Args:
            name (str): The name of the environment variable.
            default (optional): The default value to return if 
                the environment variable is not found.
            typecast (type, optional): The variable type into which to cast 
                the value.

        Returns:
            The value of the environment variable or the default value 
                if provided.

        Raises:
            ValueError: If the environment variable is missing and no default 
                is provided.
        """
        value = os.environ.get(name, default)
        if value is None:
            raise ValueError(f'Missing environment variable: {name}')

        if typecast is not None:
            try:
                return typecast(value)
            except TypeError as e:
                logger.error(
                    f'Error casting {value} as {typecast}. '
                    f'Returning {value} as {type(value)}.'
                )

        return value


# Initialize the logger for this module.
logger = get_logger(__name__)

# Load the application configuration using the EnvConfig class.
config = EnvConfig()

# Log the loaded configuration values for debugging purposes.
logger.debug(f'Using {config.OLLAMA_MODEL_NAME=}')
logger.debug(f'Endpoint: {config.OLLAMA_URL=}')
logger.debug(f'Generate: {config.OLLAMA_CHAT_URL=}')
logger.debug(f'Timeout: {config.OLLAMA_TIMEOUT=}')

# Inform that Ollama is being set up with the specified model name.
logger.info(f"Setting up ollama with {config.OLLAMA_MODEL_NAME}")

# Initialize the OllamaChatGenerator with the loaded configurations.
llm = OllamaChatGenerator(
    model=config.OLLAMA_MODEL_NAME,
    url=config.OLLAMA_CHAT_URL,
    timeout=config.OLLAMA_TIMEOUT  # Ensure the timeout value is an integer.
)
