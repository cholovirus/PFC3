from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config.env import API_KEY , GEMINI_API_KEY

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
# Inicializa el modelo
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,  # O usa la variable de entorno
    temperature=0.7,

)

messages = [
    (
        "system",
        "hola",
    ),
    ("human", "como estas"),
]

ai_msg = llm.invoke(messages)
print("gemini",ai_msg.content)
try:

    llm = ChatOpenAI(
        openai_api_key="not-needed",  # No se requiere una clave API real
        base_url="http://localhost:1234/v1"  # URL del servidor local de LM Studio
    )


    ai_msg = llm.invoke(messages)
    print("local",ai_msg.content)
except :
    print("error")


'''prompt = ChatPromptTemplate.from_template("Dime un resumen sobre {tema}")

chain = LLMChain(llm=llm, prompt=prompt)

respuesta = chain.run(tema="la Segunda Guerra Mundial")
print(respuesta)'''