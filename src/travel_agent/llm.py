from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    "google_genai:gemini-3.5-flash",
    max_retries=10,
    timeout=120,
    temperature=0.2
)

