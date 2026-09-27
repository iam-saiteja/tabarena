import os
import time
import json
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul

API_KEY = "apikey_215727f40654ed954abf9b594b8bddfa2b42_f301f9a2a5ff2c357e5623c574a3b66185b36e3bce20dc6a5f054bff6b14816e"
os.environ["TYPESAFE_API_KEY"] = API_KEY

client = TypeSafeClient(api_key=API_KEY)

print("=" * 70)
print("TESTING TYPESAFE AI JEV SYSTEM ONE MODEL")
print("=" * 70)

# Test 1: Connectivity and Model Metadata
try:
    models = client.models.list()
    print(f"[+] Available Models: {[m.id for m in models.models]}")
except Exception as e:
    print(f"[*] Model list check: {e}")

# Test 2: Structured Tabular Decision Task (Credit Risk Evaluation)
print("\n[+] Running Structured Tabular Credit Assessment Benchmark...")

test_cases = [
    {
        "id": "Profile_1_Prime",
        "state": {
            "age": 42,
            "annual_income_usd": 125000,
            "employment_length_years": 12,
            "debt_to_income_ratio": 0.18,
            "credit_score": 780,
            "delinquencies_2yr": 0,
            "loan_amount_requested": 15000,
            "purpose": "Home improvement",
        },
        "expected": "Low Risk / Approve"
    },
    {
        "id": "Profile_2_Subprime",
        "state": {
            "age": 24,
            "annual_income_usd": 32000,
            "employment_length_years": 0.5,
            "debt_to_income_ratio": 0.48,
            "credit_score": 570,
            "delinquencies_2yr": 3,
            "loan_amount_requested": 20000,
            "purpose": "Debt consolidation",
        },
        "expected": "High Risk / Reject"
    },
    {
        "id": "Profile_3_Borderline",
        "state": {
            "age": 31,
            "annual_income_usd": 68000,
            "employment_length_years": 4,
            "debt_to_income_ratio": 0.32,
            "credit_score": 665,
            "delinquencies_2yr": 1,
            "loan_amount_requested": 12000,
            "purpose": "Auto repair",
        },
        "expected": "Medium Risk / Review"
    }
]

questions = {
    "risk_tier": Choice(
        instructions="Classify the credit risk category for this loan applicant.",
        criteria={
            "low": "Strong credit profile, low DTI, stable employment, low probability of default.",
            "medium": "Acceptable credit with minor debt burden or moderate DTI, requires standard review.",
            "high": "High DTI, multiple delinquencies, low credit score, or high default risk."
        }
    ),
    "credit_health_score": Score(
        instructions="Score overall financial health from 1 (severe default risk) to 10 (exceptional creditworthiness).",
        criteria=[f"Tier {i}" for i in range(1, 11)]
    ),
    "approve_loan": Noul(
        instructions="Should this loan be automatically approved based on standard underwriting thresholds?"
    )
}

latencies = []
results_summary = []

for case in test_cases:
    t0 = time.perf_counter()
    response = client.system_one(
        state=case["state"],
        questions=questions
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    latencies.append(latency_ms)

    risk_ans = response.answers["risk_tier"]
    score_ans = response.answers["credit_health_score"]
    approve_ans = response.answers["approve_loan"]

    res = {
        "id": case["id"],
        "latency_ms": f"{latency_ms:.1f}ms",
        "risk_choice": risk_ans.choice,
        "risk_confidence": f"{risk_ans.confidence:.3f}" if hasattr(risk_ans, "confidence") else "N/A",
        "health_score": score_ans.score,
        "approve": approve_ans.noul,
        "approve_prob": f"{approve_ans.probability:.3f}" if hasattr(approve_ans, "probability") and approve_ans.probability is not None else "N/A"
    }
    results_summary.append(res)
    
    print(f"\n--- {case['id']} (Latency: {latency_ms:.1f}ms) ---")
    print(f"  * Risk Tier Decision: {risk_ans.choice}")
    if hasattr(risk_ans, "probabilities") and risk_ans.probabilities:
        print(f"    Probabilities: {risk_ans.probabilities}")
    print(f"  * Credit Health Score (1-10): {score_ans.score}")
    print(f"  * Approve (Yes/No): {approve_ans.noul}")

print("\n" + "=" * 70)
print(f"AVERAGE JEV SYSTEM ONE LATENCY: {sum(latencies)/len(latencies):.1f} ms")
print("=" * 70)
