from langchain_ollama import ChatOllama
from openai import OpenAI


def main(model: str, api_key: str = ""):

    if model.lower() == "llama":
        client = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

    elif model.lower() in ["openai", "gpt"]:
        client = OpenAI(api_key=api_key)

    else:
        msg = f"Expected model to be either llama, openai or gpt. Got {model}."
        raise ValueError(msg)

    return client
