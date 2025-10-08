# langextract-bedrock

AWS Bedrock provider plugin for [LangExtract](https://github.com/google/langextract).  
This package adds native support for Anthropic Claude, Mistral, Cohere, Meta Llama, and Amazon Titan models through AWS Bedrock — enabling scalable structured entity and relationship extraction directly from text.

---

## 🚀 Installation



### From GitHub (latest)
```bash
pip install "git+https://github.com/vjhawar-sp/langextract-bedrock.git"
```

Requires **Python ≥ 3.9**

Install additional dependencies:
```bash
pip install langextract boto3 botocore python-dotenv
```


---

## ⚙️ Environment Setup

Create a `.env` file in your project root:
```bash
AWS_PROFILE=your-aws-profile
AWS_REGION=us-west-2
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

Verify your AWS credentials:
```bash
aws sts get-caller-identity --profile your-aws-profile
```

---

## 🧠 Quick Start Example

```python
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
        "temperature": 0.2,
        "max_output_tokens": 256,
    },
)

model = factory.create_model(cfg)

for outputs in model.infer(["Quick connectivity check. Reply with OK."]):
    print(outputs[0].output)
```

---

## 🧩 Structured Extraction Example

```python
import os, json
from dotenv import load_dotenv
from langextract import factory

load_dotenv()

schema = {
  "type": "object",
  "properties": {
    "entities": {"type": "array"},
    "relationships": {"type": "array"}
  },
  "required": ["entities", "relationships"]
}

cfg = factory.ModelConfig(
    model_id=f"bedrock:{os.getenv('BEDROCK_MODEL_ID')}",
    provider="BedrockLanguageModel",
    provider_kwargs={
        "region_name": os.getenv("AWS_REGION"),
        "aws_profile": os.getenv("AWS_PROFILE"),
        "response_schema": schema,
        "structured_output": True,
    },
)

model = factory.create_model(cfg)

prompt = (
  "Extract entities and relationships from the following:\n"
  "Officer Lee recovered a Glock 19 and four casings near 5th & Pine. "
  "Witness Dana Chen saw Alex Rivera discard the weapon."
)

for outputs in model.infer([prompt]):
    data = json.loads(outputs[0].output)
    print(json.dumps(data, indent=2))
```

---

## 📄 JSONL Export Example

See `examples/run_extract_entities.py` for a complete multi-document workflow demonstrating:
- Extraction rules and schema definitions  
- Batch processing via AWS Bedrock  
- Export of structured entities to `entities_export.jsonl`

---

## 🧰 Development Setup

```bash
git clone https://github.com/vaibhavijhawar/langextract-bedrock.git
cd langextract-bedrock
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify that LangExtract detects your Bedrock provider:
```python
import langextract as lx
lx.providers.load_plugins_once()
print(lx.providers.registry.list_entries())
```

---

## 👩‍💻 Maintainer

**Vaibhavi Jhawar**  
Licensed under the Apache License 2.0.

---

## 🪄 Future Plans

- Streaming response support  
- Bedrock Claude JSON schema enforcement  
- Async batch inference for large-scale document ingestion  

---
