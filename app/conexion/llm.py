from openai import OpenAI
import google.generativeai as genai

import os
import re

class MultiLLM:
    def __init__(self, openai_key=None, gemini_key=None, lmstudio_url="http://localhost:1234/v1/chat/completions"):
        """Inicializa las APIs con sus claves y la URL local de LM Studio."""
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.lmstudio_url = OpenAI(base_url=lmstudio_url, api_key="lm-studio")

        
        # Configurar OpenAI
        OpenAI.api_key = self.openai_key

        # Configurar Gemini
        genai.configure(api_key=self.gemini_key)

    def chat_openai(self, prompt, model="gpt-4o-mini"):
        """Llama a OpenAI GPT."""
        try:
            response = OpenAI.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error en OpenAI: {e}"

    def chat_gemini(self, prompt, model="gemini-1.5-flash-latest"):
        """Llama a Gemini (Google AI)."""
        try:
            model = genai.GenerativeModel(model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error en Gemini: {e}"

    def chat_lmstudio(self, prompt, model="model-identifier"):
        """Llama a LM Studio (modelo local en la API de OpenAI)."""
        try:
            completion = self.lmstudio_url.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            message_content = completion.choices[0].message.content
            cleaned_content = re.sub(r'<think>.*?</think>', '', message_content, flags=re.DOTALL)
 
            return cleaned_content.strip()
           
        except Exception as e:
            return f"Error en LM Studio: {e}"


