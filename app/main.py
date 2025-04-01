
from conexion.db_vector import *
from config.env import API_KEY , GEMINI_API_KEY
from conexion.llm import *
# Insertar usando tu clase

records = [
    {
        "id": "323",
        "chunk_text": "La diabetes es una enfermedad crónica",
        "metadata": {
            "categoria": "endocrinologia",
            "autor": "Dr. Perez",
            "año": 2022
        }
    },
    {
        "id": "doc3232",
        "chunk_text": "El infarto de miocardio requiere atención inmediata",
        "metadata": {
            "categoria": "cardiologia",
            "autor": "Dra. Gomez",
            "año": 2023
        }
    }
]


LLM = MultiLLM(gemini_key=GEMINI_API_KEY,lmstudio_url="http://localhost:1234/v1")
promt = "hola como estas"
#print(LLM.chat_gemini(promt))
print(LLM.chat_lmstudio(promt))
'''
client = PineconeClient(api_key=API_KEY, index_name="pfc3")
records =client.prepare_records(records)
#client.upsert_records(namespace="test", records=records)

results = client.search_by_metadata(
    query_="instrument violin",
    filter={
        "category": {"$eq": "music"},
    },
    top_k=10,
    namespace="medicina"
)
print(results)'
'''
