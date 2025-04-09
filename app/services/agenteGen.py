import json
import os
import random
from faker import Faker
import concurrent.futures
from services.prompt import *
from langchain_core.messages import HumanMessage, SystemMessage
fake = Faker("es_ES")



def generar_agentes(n, habilidades,personalidades, LLM):
    """Genera n agentes con características detalladas y los guarda en archivos JSON."""
    
    prompt_contexto = prompt_contexto_
    prompt_opinion = prompt_opinion_
    prompt_forma_hablar = prompt_opinion_
    
    
    os.makedirs("initial_agents", exist_ok=True)
    
    for i in range(n):
        print(i)
        nombre = fake.name()
        alias = fake.user_name()
        personalidad = random.choice(personalidades)
        habilidad = random.sample(habilidades, k=random.randint(2, 4))

        # Personalizar los prompts con el nombre, personalidad y habilidades
        prompt_contexto_personalizado = prompt_contexto.format(nombre=nombre, personalidad=personalidad, habilidades=", ".join(habilidad))
        prompt_opinion_personalizado = prompt_opinion.format(nombre=nombre, personalidad=personalidad, habilidades=", ".join(habilidad))
        prompt_forma_hablar_personalizado = prompt_forma_hablar.format(nombre=nombre, personalidad=personalidad, habilidades=", ".join(habilidad))

        # Crear los mensajes de LangChain
        system_message = SystemMessage(content="Por favor, genera una respuesta concisa con un máximo de 100 tokens para cada mensaje. No te extiendas demasiado.")
        mensaje_contexto = [system_message , HumanMessage(content=prompt_contexto_personalizado)]
        mensaje_opinion = [system_message,HumanMessage(content=prompt_opinion_personalizado)]
        mensaje_forma_hablar = [system_message, HumanMessage(content=prompt_forma_hablar_personalizado)]

        # Hacer las peticiones de manera secuencial (eliminamos ThreadPoolExecutor)
        contexto = LLM.invoke(mensaje_contexto).content
        opinion_sobre_si_mismo = LLM.invoke(mensaje_opinion).content
        forma_de_hablar = LLM.invoke(mensaje_forma_hablar).content
        

        # Crear y guardar el agente en un archivo JSON
        agente = {
            "nombre": nombre,
            "alias": alias,
            "personalidad": personalidad,
            "habilidades": habilidad,
            "contexto": contexto,
            "opinion_sobre_si_mismo": opinion_sobre_si_mismo,
            "forma_de_hablar": forma_de_hablar
        }
        
        # Guardar el agente en un archivo JSON
        with open(f"initial_agents/agent_{i+1}.json", "w", encoding="utf-8") as f:
            json.dump(agente, f, indent=4, ensure_ascii=False)

    print(f"Se generaron {n} agentes en la carpeta 'initial_agents/'")
    
    
    