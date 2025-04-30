# fact_verification_system.py

from transformers import pipeline
from extract import extract_claims
from retriever import retrieve_evidence

# NLI Verifier

nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")

def run_nli(premise, hypothesis):
    """
    NLI model: premise = evidence, hypothesis = claim
    """
    result = nli_pipeline(f"{premise} </s> {hypothesis}", truncation=True)[0]
    return {
        "label": result["label"],
        "score": result["score"]
    }

def aggregate_nli_verdicts(claim, evidences, k=3):
    """
    Do NLI over multiple evidences and aggregates verdicts.
    Output the best label without thresholding.
    """
    verdict_scores = {"ENTAILMENT": 0, "CONTRADICTION": 0, "NEUTRAL": 0}

    for evidence in evidences[:k]:
        nli_result = run_nli(premise=evidence, hypothesis=claim)
        label = nli_result["label"].upper()
        score = nli_result["score"]
        verdict_scores[label] += score

    final_label = max(verdict_scores.items(), key=lambda x: x[1])[0]
    final_score = verdict_scores[final_label] / k

    return {
        "label": final_label,
        "confidence": round(final_score, 4)
    }

# Full Fact Verification

def verify_claims_from_text(input_text):
    claims = extract_claims(input_text)
    results = []

    for claim in claims:
        evidences = retrieve_evidence(claim)
        verdict = aggregate_nli_verdicts(claim, evidences, k=min(3, len(evidences)))

        results.append({
            "claim": claim,
            "verdict": verdict,  
            "evidences": evidences
        })

    return results

# === Example ===


if __name__ == "__main__":
    input_text = """
    The White House is in China. Buzz Aldrin was the first man to walk on the moon. 
    Mount Everest is in Saudi Arabia. Arizona is the sunniest state in the United States.
    """

    verification_results = verify_claims_from_text(input_text)

    for result in verification_results:
        print("\n---")
        print(f"Claim: {result['claim']}")
        print(f"Verdict: {result['verdict']['label']} (confidence: {result['verdict']['confidence']})")
        '''print("Top Evidences:")
        for i, evidence in enumerate(result['evidences'], 1):
            print(f"{i}. {evidence}")'''
