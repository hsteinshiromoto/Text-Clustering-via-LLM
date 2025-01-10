from langchain_ollama import ChatOllama
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import os


def main(model: str) -> Union[ChatOllama, OpenAI, Anthropic]:
    """Initialize and return an AI model client based on the specified model type.

    This function loads the necessary API keys from environment variables and
    initializes the requested AI model client. Supports local Llama model,
    OpenAI's GPT models, and Anthropic's Claude models.

    Args:
        model (str): The type of model to initialize. Valid options are:
            - "llama": For local Llama model
            - "openai" or "gpt": For OpenAI's GPT models
            - "claude" or "anthropic": For Anthropic's Claude models

    Returns:
        Union[ChatOllama, OpenAI, Anthropic]: An initialized client for the specified model.

    Raises:
        ValueError: If an invalid model type is specified or if required API keys
            are not found in environment variables.

    Examples:
        >>> client = main("llama")  # Initialize Llama model
        >>> client = main("openai")  # Initialize OpenAI model (requires OPENAI_API_KEY in .env)
        >>> client = main("claude")  # Initialize Claude model (requires ANTHROPIC_API_KEY in .env)
    """

    load_dotenv()

    if model.lower() == "llama":
        client = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

    elif model.lower() in ["openai", "gpt"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=api_key)

    elif model.lower() in ["claude", "anthropic"]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        client = Anthropic(api_key=api_key)

    else:
        msg = f"Expected model to be either llama, openai/gpt, or claude/anthropic. Got {model}."
        raise ValueError(msg)

    return client
