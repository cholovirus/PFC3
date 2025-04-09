
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from typing import Literal
import os
import re

class MultiLLM:
    def __init__(self, openai_key=None, gemini_key=None, lmstudio_url="http://localhost:1234/v1"):
        self.openai_key = openai_key
        self.gemini_key = gemini_key
        self.lmstudio_url = lmstudio_url

    def get_model(self, provider: Literal["openai", "gemini", "lmstudio"], model_name=None, temperature=0.5,max_tokens_=1000):
        if provider == "openai":
            return ChatOpenAI(
                openai_api_key=self.openai_key,
                model=model_name or "gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens_,
                stop=[ ".", "!", "?"]
            )

        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-1.5-flash",
                google_api_key=self.gemini_key,
                temperature=temperature,
                max_tokens=max_tokens_,
                stop=[ ".", "!", "?"]
            )

        elif provider == "lmstudio":
            return ChatOpenAI(
                openai_api_key="not-needed",  # LM Studio no requiere clave
                base_url=self.lmstudio_url,
                model=model_name or "lmstudio-model",  # reemplaza por el identificador de tu modelo local
                temperature=temperature,
                max_tokens=max_tokens_,
                stop=[ ".", "!", "?"]
            )

        else:
            raise ValueError("Proveedor no soportado")


