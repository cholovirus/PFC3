from dotenv import load_dotenv # type: ignore
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'

load_dotenv(dotenv_path=env_path,override=True)

API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
