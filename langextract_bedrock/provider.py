import json
import os
import re
from typing import Iterable, List, Dict, Any, Generator

import boto3
import langextract as lx
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput
# Examples of model id patterns Bedrock hosts (vendor prefixes optional in your UX):
#   anthropic.claude-3-5-sonnet-20240620-v1:0
#   mistral.mistral-large-2407-v1:0
#   cohere.command-r-plus-v1:0
#   meta.llama3-8b-instruct-v1:0
#   amazon.titan-text-lite-v1
#
# We’ll accept either:
#   model_id="bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0"
# or just the raw Bedrock model ID; patterns below catch both.

@lx.providers.registry.register(
    r'^bedrock:',                     # explicit "bedrock:" prefix
    r'^(anthropic|mistral|cohere|meta\.llama|amazon\.titan)',  # bare Bedrock IDs
    priority=10
)
class BedrockLanguageModel(BaseLanguageModel):
    """
    LangExtract provider for AWS Bedrock via boto3 `bedrock-runtime`.
    Supports multiple vendors by inspecting the model_id prefix.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str = None,
        aws_profile: str = None,
        # generic text-gen kwargs
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.9,
        # schema knobs passed by LangExtract (soft-enforced here)
        response_schema: Dict[str, Any] = None,
        structured_output: bool = False,
        **kwargs
    ):
        super().__init__()
        # normalize model id (strip optional "bedrock:" prefix)
        self.raw_model_id = re.sub(r'^bedrock:', '', model_id)

        # Auth / client
        session_kwargs = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", region_name=region_name)

        self.temperature = temperature
        self.max_tokens = max_output_tokens
        self.top_p = top_p

        self.response_schema = response_schema
        self.structured_output = structured_output

        # detect vendor branch
        if self.raw_model_id.startswith("anthropic."):
            self._vendor = "anthropic"
        elif self.raw_model_id.startswith("mistral."):
            self._vendor = "mistral"
        elif self.raw_model_id.startswith("cohere."):
            self._vendor = "cohere"
        elif self.raw_model_id.startswith("meta.llama"):
            self._vendor = "llama"
        elif self.raw_model_id.startswith("amazon.titan"):
            self._vendor = "titan"
        else:
            # fallback: try to infer from model name tokens
            self._vendor = "generic"

    @classmethod
    def get_schema_class(cls):
        """
        Optional: Tell LangExtract we can accept a schema object.
        We don’t enforce at the model level (Bedrock JSON schema
        varies by vendor). We’ll post-validate in Python as needed.
        """
        # If you’ve built a custom Schema, return it here.
        # Returning None tells LangExtract we’ll accept `response_schema`
        # via kwargs but don’t construct it ourselves.
        return None

    def _prompt_for_schema(self, prompt: str) -> str:
        """
        If structured_output=True but vendor lacks native JSON schema,
        we steer with instruction + ask for strict JSON.
        """
        if self.structured_output and self.response_schema:
            return (
                "Return ONLY valid JSON that matches this schema. "
                "Do not include explanations.\n"
                f"Schema (JSON Schema-ish hint): {json.dumps(self.response_schema) }\n\n"
                f"Task:\n{prompt}"
            )
        return prompt

    def _invoke_anthropic(self, prompt: str) -> str:
        """
        Bedrock Anthropic Messages format.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        resp = self.client.invoke_model(
            modelId=self.raw_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        # Anthropic returns list of content blocks; pick the first text
        parts = payload.get("content", [])
        text = ""
        for p in parts:
            if p.get("type") == "text":
                text += p.get("text", "")
        return text.strip()

    def _invoke_mistral(self, prompt: str) -> str:
        """
        Bedrock Mistral format (chat).
        """
        body = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        resp = self.client.invoke_model(
            modelId=self.raw_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        # Mistral on Bedrock returns {"outputs": [{"text": "..."}], ...}
        outputs = payload.get("outputs", [])
        return (outputs[0].get("text", "") if outputs else "").strip()

    def _invoke_generic(self, prompt: str) -> str:
        """
        Fallback for other Bedrock text models (Cohere, Llama, Titan).
        Most accept a similar request with 'prompt' or 'inputText'.
        You can extend with vendor-specific branches as needed.
        """
        # try generic "prompt"
        body = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        resp = self.client.invoke_model(
            modelId=self.raw_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        # Cohere: {"generations":[{"text": "..."}]}
        if "generations" in payload and payload["generations"]:
            return payload["generations"][0].get("text", "").strip()
        # Titan / Llama often: {"results":[{"outputText":"..."}]} or {"outputs":[{"text":"..."}]}
        if "results" in payload and payload["results"]:
            return payload["results"][0].get("outputText", "").strip()
        if "outputs" in payload and payload["outputs"]:
            return payload["outputs"][0].get("text", "").strip()
        # last resort
        return payload.get("outputText", "").strip()

    def infer(self, batch_prompts: Iterable[str], **kwargs) -> Generator[List[ScoredOutput], None, None]:
        """
        LangExtract contract: yield a list of ScoredOutput per prompt.
        We don’t stream here; you can add streaming later if needed.
        """
        for p in batch_prompts:
            prompt = self._prompt_for_schema(p)
            if self._vendor == "anthropic":
                text = self._invoke_anthropic(prompt)
            elif self._vendor == "mistral":
                text = self._invoke_mistral(prompt)
            else:
                text = self._invoke_generic(prompt)

            # Optional: strict post-validation if structured_output=True
            if self.structured_output and self.response_schema:
                try:
                    # Minimal: ensure it’s valid JSON; enforce your own schema validator if desired
                    parsed = json.loads(text)
                    # You can integrate jsonschema or pydantic validation here.
                except Exception:
                    # If invalid JSON, wrap in best-effort object
                    text = json.dumps({"_raw": text, "_error": "Invalid JSON for requested schema"})

            yield [ScoredOutput(score=1.0, output=text)]
