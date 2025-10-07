# langextract_bedrock/__init__.py
import os
from dotenv import load_dotenv

# Load your .env automatically when the plugin loads
load_dotenv()

# Default environment fallbacks for Bedrock region/profile
os.environ.setdefault("AWS_REGION", os.getenv("AWS_REGION", "us-west-2"))
os.environ.setdefault("AWS_PROFILE", os.getenv("AWS_PROFILE", None))

# Export the provider class so LangExtract's entry point can find it
from .provider import BedrockLanguageModel
