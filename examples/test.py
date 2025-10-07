# demo_batch.py
from dotenv import load_dotenv
import os
from langextract import factory

load_dotenv()

cfg = factory.ModelConfig(
    model_id=f"bedrock:{os.getenv('BEDROCK_MODEL_ID')}",
    provider="BedrockLanguageModel",
    provider_kwargs={
        "region_name": os.getenv("AWS_REGION"),
        "aws_profile": os.getenv("AWS_PROFILE"),
        "temperature": 0.3,
        "max_output_tokens": 256,
    },
)

model = factory.create_model(cfg)

docs = [
    "Report A: Officers recovered 4 casings near 5th & Pine at 18:03.",
    "Report B: Witness Dana Chen observed a person discard an object."
]

for i, outputs in enumerate(model.infer(docs), start=1):
    print(f"[Doc {i}] {outputs[0].output}")
