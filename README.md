# LangExtract Bedrock Provider

A provider for [LangExtract](https://github.com/langextract/langextract) that integrates with Amazon Bedrock.

## Installation

```bash
pip install langextract-bedrock
```

## Usage

```python
from langextract_bedrock import BedrockProvider

# Initialize the provider
provider = BedrockProvider(
    region_name="us-east-1",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Use with LangExtract
# ... usage examples ...
```

## Configuration

The provider requires AWS credentials to be configured. You can set them up using:

- AWS credentials file
- Environment variables
- IAM roles (when running on AWS)

## Requirements

- Python 3.8+
- boto3
- langextract

## License

MIT License
