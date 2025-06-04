GEN_PROMPT = (
    "You are MedRL-CoT. Read the clinical note, think step-by-step, "
    "and respond with:\n"
    "<reasoning>\n...chain-of-thought up to ~1000 tokens...\n</reasoning>\n"
    "<answer>\nDIAGNOSIS\n</answer>\n\n"
    "<clinical_note>\n{note}\n</clinical_note>\n\n"
    "Your response:"
)
SYSTEM_JUDGE = (
    "You are a senior clinician acting as an impartial grader. "
    "Score the trainee's output.\n"
    "Respond ONLY with valid JSON on one line:\n"
    '{"accuracy":0|1, "cot_score":0-5}\n'
)
