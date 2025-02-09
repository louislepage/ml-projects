import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fetch API key
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or API_KEY == "":
    raise ValueError("ERROR: OPENAI_API_KEY is not set!")

# Export API_KEY so other modules can use it
__all__ = ["API_KEY"]