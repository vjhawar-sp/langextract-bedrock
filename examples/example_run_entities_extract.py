# run_extract_entities.py
# Usage:
#   python run_extract_entities.py
#
# Creates: outputs/entities_export.jsonl

from dotenv import load_dotenv
import os, json, pathlib, re
from langextract import factory
import langextract as lx

# -----------------------------
# 0) Setup
# -----------------------------
load_dotenv()
OUT_DIR = pathlib.Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_PATH = OUT_DIR / "entities_export.jsonl"

# Optional: show registry (helps debug plugin discovery)
lx.providers.load_plugins_once()
print("Registry entries:", lx.providers.registry.list_entries())

# -----------------------------
# 1) Extraction Rules (High-level)
# -----------------------------
RULES = """
You are an information extraction agent. Follow these STRICT rules:

1) Output MUST be valid JSON (no markdown) matching the provided schema.
2) Extract ENTITIES with fields:
   - id: stable string (e.g., "E1", "E2", ...)
   - type: one of {Person, Organization, Location, Address, Vehicle, Weapon, Event, Evidence, Time, Date, Phone, Email, Other}
   - name: canonical surface form
   - attributes: optional free-form key→value map (e.g., { "age": "34", "role": "Witness" })

3) Extract RELATIONSHIPS with fields:
   - source: an entity id (e.g., "E1")
   - target: an entity id (e.g., "E5")
   - type: one of {
       "occurred_at", "resides_at", "affiliated_with", "works_for", "calls", "owns",
       "uses", "wields", "travels_in", "located_near", "reports", "observed",
       "involved_in", "evidence_of", "time_of", "date_of", "contact_of", "aka"
     }
   - attributes: optional map (e.g., confidence scores or qualifiers)

4) Disambiguation & consistency:
   - Merge duplicate mentions into one canonical entity.
   - Prefer precise entity types (Address vs Location when a street is given).
   - Use “Event” only for concrete incidents/actions (e.g., “traffic stop”, “shots fired”).
   - Preserve spans with uncertain info under attributes (e.g., {"uncertain": "vehicle may be black"}).

5) No hallucinations:
   - If a field is unknown, omit it. Never invent IDs or facts.
   - Every relationship must refer to existing entity ids.

6) Provenance:
   - Add "provenance": {"doc_id": <doc_id>} at the top level of the final JSON object.

Return ONLY valid JSON.
"""

# -----------------------------
# 2) JSON Schema hint (soft constraints)
# -----------------------------
SCHEMA_HINT = {
  "type": "object",
  "properties": {
    "entities": {"type": "array", "items": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "name": {"type": "string"},
        "attributes": {"type": "object"}
      },
      "required": ["id", "type", "name"]
    }},
    "relationships": {"type": "array", "items": {
      "type": "object",
      "properties": {
        "source": {"type": "string"},
        "target": {"type": "string"},
        "type": {"type": "string"},
        "attributes": {"type": "object"}
      },
      "required": ["source", "target", "type"]
    }},
    "provenance": {"type": "object"}  # we'll inject doc_id post-parse if needed
  },
  "required": ["entities", "relationships"]
}

# -----------------------------
# 3) Example “larger” document set
#    (replace with your real data; can be thousands of chars)
# -----------------------------
DOCS = [
    {
        "doc_id": "CV-IR-2025-10-05-001",
        "text": (
            "On Oct 5, 2025 at approximately 18:03, officers responded to reports of shots fired "
            "near 5th & Pine in Chula Vista. Witness Dana Chen stated she observed an individual, "
            "later identified as Alex Rivera, discard a 9mm handgun behind the deli at 512 Pine St. "
            "Officer Lee recovered four 9mm casings and a black Glock 19 near the location. "
            "A grey Honda Civic (CA plate 7ABC123) was seen leaving eastbound on Pine. "
            "Rivera is affiliated with Southside Athletics, works part-time at Mercado Deli, "
            "and is believed to reside at 900 Maple Ave. No victims located. A second witness, "
            "Michael Ortiz, reported hearing five shots and seeing Rivera place the handgun in a trash bin."
        )
    },
    {
        "doc_id": "CV-IR-2025-10-05-002",
        "text": (
            "At about 18:15, Officer Patel conducted a traffic stop of a grey Honda Civic matching the description. "
            "Driver identified as Alex R.; registration lists owner as Alicia Rivera residing at 900 Maple Ave, Chula Vista. "
            "No firearm recovered in the vehicle. Body-worn camera recorded the stop; timestamp aligns with initial incident. "
            "Witness phone number for follow-up: (619) 555-0138."
        )
    }
]

# -----------------------------
# 4) Build prompts (rules + text)
# -----------------------------
def build_prompt(doc):
    return (
        f"{RULES}\n\n"
        f"Document (doc_id={doc['doc_id']}):\n"
        f"{doc['text']}\n\n"
        f"Return JSON now."
    )

PROMPTS = [build_prompt(d) for d in DOCS]

# -----------------------------
# 5) Configure Bedrock via factory
# -----------------------------
cfg = factory.ModelConfig(
    model_id=f"bedrock:{os.getenv('BEDROCK_MODEL_ID')}",
    provider="BedrockLanguageModel",
    provider_kwargs={
        "region_name": os.getenv("AWS_REGION"),
        "aws_profile": os.getenv("AWS_PROFILE"),
        "temperature": 0.1,             # conservative for extraction
        "max_output_tokens": 1024,
        "response_schema": SCHEMA_HINT, # steer toward JSON
        "structured_output": True
    },
)
model = factory.create_model(cfg)

# -----------------------------
# 6) Run inference
# -----------------------------
print("Running extraction...")
results = []
for doc, outputs in zip(DOCS, model.infer(PROMPTS)):
    raw = outputs[0].output
    # Best-effort JSON parse; if not valid JSON, wrap as _raw
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"entities": [], "relationships": [], "_raw": raw}

    # Ensure provenance included
    parsed.setdefault("provenance", {})
    parsed["provenance"]["doc_id"] = doc["doc_id"]

    # Optional: quick normalization (dedupe entity ids)
    seen_ids = set()
    norm_entities = []
    for ent in parsed.get("entities", []):
        eid = ent.get("id")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            norm_entities.append(ent)
    parsed["entities"] = norm_entities

    # Optional: filter relationships to existing entity ids
    valid_ids = {e.get("id") for e in parsed["entities"] if e.get("id")}
    norm_rels = []
    for rel in parsed.get("relationships", []):
        if rel.get("source") in valid_ids and rel.get("target") in valid_ids:
            norm_rels.append(rel)
    parsed["relationships"] = norm_rels

    results.append(parsed)

# -----------------------------
# 7) Write JSONL export
# -----------------------------
with open(EXPORT_PATH, "w", encoding="utf-8") as f:
    for obj in results:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Wrote {len(results)} objects to {EXPORT_PATH}")
