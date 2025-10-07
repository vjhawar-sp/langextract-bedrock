# langextract-bedrock

**AWS Bedrock provider plugin for [LangExtract](https://github.com/google/langextract)**  
Adds native support for Anthropic, Mistral, Cohere, Llama, and Titan models hosted on Amazon Bedrock.

---

## 🚀 Features

- 🔌 Seamless integration with LangExtract’s provider registry  
- ☁️ Supports multiple Bedrock vendors (Anthropic Claude, Mistral, Cohere, Meta Llama, Amazon Titan)  
- 🔒 Automatic environment loading via `.env` (no code changes required)  
- 🧩 Schema-aware extraction (structured JSON or free-form)  
- 🪄 Compatible with LangExtract ≥ 1.0.0 using the new `factory.ModelConfig` API  

---

## 📦 Installation

### From PyPI (when published)
```bash
pip install langextract-bedrock
