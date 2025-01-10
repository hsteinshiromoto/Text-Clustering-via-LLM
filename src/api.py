from langchain_ollama import ChatOllama
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import os


def main(model: str):
    # Load environment variables
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
