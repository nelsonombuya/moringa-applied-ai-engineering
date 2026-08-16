# Reuse imports/client/raw_patient_sms from Lab 13.

extraction_prompt = """
You are a backend administrative data extraction engine for AfyaPlus Health.
Analyse the following untrusted user SMS text. Return a valid JSON object
matching this schema:
{
  "patient_age_years": integer or null,
  "symptoms": ["string", "string"],
  "location_cluster": "string",
  "requires_emergency_dispatch": boolean,
  # TODO 1: Add "severity_score": integer 1-10 here.
}
# TODO 2: Add a scoring rule: 1-3 mild, 4-7 moderate, 8-10 severe.
CRITICAL: Return ONLY raw JSON, no markdown.
"""

# ... same API call as Lab 13 ...
parsed = json.loads(response.choices[0].message.content)
print(json.dumps(parsed, indent=2))
# TODO 3: After parsing, print "HIGH SEVERITY ..." when severity_score >= 8.
