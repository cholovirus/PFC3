import json
import os
import random
from faker import Faker
import concurrent.futures

fake = Faker("es_ES")

def generar_agentes(n, roles, habilidades,personalidades, LLM):
    """Genera n agentes con características detalladas y los guarda en archivos JSON."""
    
    prompt_contexto = (
        "Imagina que eres una persona con una vida compleja y llena de experiencias. "
        "Tu nombre es {nombre}, tienes la personalidad {personalidad} y tus habilidades incluyen {habilidades}. "
        "Cuéntame sobre tu vida como si estuvieras escribiendo en un diario personal. "
        "Incluye detalles sobre tu infancia, momentos felices, desafíos difíciles, personas importantes en tu vida y cómo te convirtiste en quien eres hoy. "
        "Habla de tus mayores logros y fracasos, cómo te hicieron sentir y qué aprendiste de ellos. "
        "Usa un tono cercano y emotivo, como si estuvieras recordando momentos clave de tu vida con un amigo de confianza. "
        "Añade detalles únicos que te hagan especial."
    )

    prompt_opinion = (
        "Reflexiona profundamente sobre quién eres. "
        "Tu nombre es {nombre}, tu personalidad es {personalidad} y tus habilidades son {habilidades}. "
        "Si te miraras al espejo y tuvieras que describirte a alguien que nunca te ha conocido, ¿qué dirías? "
        "Habla de lo que más te gusta de ti mismo, pero también de lo que quisieras cambiar. "
        "¿Tienes miedos, inseguridades o cosas que ocultas a los demás? ¿Qué te motiva a seguir adelante todos los días? "
        "Escribe como si estuvieras en una conversación interna, sin censura ni filtros, con total sinceridad. "
        "Incluye anécdotas o momentos específicos que hayan moldeado tu forma de ser."
    )

    prompt_forma_hablar = (
        "Imagina que estás en una habitación con diferentes tipos de personas: amigos cercanos, desconocidos y rivales. "
        "Tu nombre es {nombre}, tienes la personalidad {personalidad} y tus habilidades son {habilidades}. "
        "Describe cómo cambia tu manera de hablar con cada uno. "
        "¿Eres relajado y bromista con tus amigos? ¿Eres formal y distante con extraños? ¿Tiendes a hablar con sarcasmo o eres más directo y pragmático? "
        "Piensa en ejemplos concretos. "
        "Si alguien te hiciera un cumplido, ¿cómo responderías? ¿Y si alguien te insultara? "
        "Explica cómo tu tono, tus palabras y tu lenguaje corporal cambian según la situación."
    )



    os.makedirs("initial_agents", exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(n):
            nombre = fake.name()
            alias = fake.user_name()
            personalidad = random.choice(personalidades)
            habilidad = random.sample(habilidades, k=random.randint(2, 4))

            # Personalizar los prompts con el nombre, personalidad y habilidades
            prompt_contexto_personalizado = prompt_contexto.format(nombre=nombre, personalidad=personalidad, habilidades=", ".join(habilidad))
            prompt_opinion_personalizado = prompt_opinion.format(nombre=nombre, personalidad=personalidad, habilidades=", ".join(habilidad))
            prompt_forma_hablar_personalizado = prompt_forma_hablar.format(nombre=nombre, personalidad=personalidad, habilidades=", ".join(habilidad))
            # Hacer las peticiones en paralelo
           
            future_contexto = executor.submit(LLM.chat_gemini, prompt_contexto_personalizado)
            future_opinion = executor.submit(LLM.chat_gemini, prompt_opinion_personalizado)
            future_forma_hablar = executor.submit(LLM.chat_gemini, prompt_forma_hablar_personalizado)
            
            # Esperar a que las respuestas lleguen
            contexto = future_contexto.result()
            opinion_sobre_si_mismo = future_opinion.result()
            forma_de_hablar = future_forma_hablar.result()
            print(i)

            # Crear el agente
            agente = {
                "id": i + 1,
                "nombre": nombre,
                "alias": alias,
                "edad": random.randint(20, 60),
                "personalidad":personalidad,
                "habilidades": habilidad,
                "rol": random.choice(roles),
                "contexto": contexto,
                "opinion_sobre_si_mismo": opinion_sobre_si_mismo,
                "forma_de_hablar": forma_de_hablar
            
            }

            # Guardar el agente en un archivo
            with open(f"initial_agents/agent_{i+1}.json", "w", encoding="utf-8") as f:
                json.dump(agente, f, indent=4, ensure_ascii=False)

    print(f"Se generaron {n} agentes en la carpeta 'initial_agents/'")
    
    