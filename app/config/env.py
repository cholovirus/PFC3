from dotenv import load_dotenv # type: ignore
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Ahora puedes acceder a las variables de entorno en cualquier parte del proyecto
# Ejemplo de cómo acceder a una variable de entorno:
API_KEY = os.getenv("PINECONE_API_KEY")

