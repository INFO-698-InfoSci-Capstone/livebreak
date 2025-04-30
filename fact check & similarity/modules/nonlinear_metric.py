import math

def compute_exp_fact_score(verification_results, alpha=1.5, lam=3.0, beta=0.5):
    """
    Computes a document-level fact score.

    Parameters:
    - alpha: scaling factor for contradiction penalty
    - lam: exponential factor for contradiction penalty
    - beta: scaling factor for neutral penalty

    Return:
    - A float score averaged across all claims
    """
    score = 0.0

    for result in verification_results:
        label = result["verdict"]["label"].upper()
        confidence = result["verdict"]["confidence"]

        if label == "ENTAILMENT":
            score += confidence  # linear reward
        elif label == "CONTRADICTION":
            score -= alpha * math.exp(lam * confidence)  # exponential penalty
        elif label == "NEUTRAL":
            score -= beta * math.log(1 + confidence)  # mild log penalty

    return round(score / len(verification_results), 4) if verification_results else 0.0