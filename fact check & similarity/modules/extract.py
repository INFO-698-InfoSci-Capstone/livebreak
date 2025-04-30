import spacy
from preprocess import preprocess_with_spacy
def extract_claims(text: str):
    """
    Extracts claim-like sentences from input text using syntax.
    
    Returns:
        List of factual claims.
    """
    sents = preprocess_with_spacy(text)
    claims = []

    for sent in sents:
        has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
        has_verb = any(tok.pos_ == "VERB" for tok in sent)
        has_entity = any(ent.label_ in ("ORG", "PERSON", "GPE", "DATE", "PRODUCT", "EVENT") for ent in sent.ents)

        if has_subject and has_verb and has_entity:
            claims.append(sent.text.strip())

    return claims

# test
sample_text = """
<p>NASA is sending astronauts back to the moon.</p>\n\n A total of 12 astronauts have gone to the moon. 
One famous quote is "One small step for man, one giant leap for mankind." <br><br>    
NASA is planning a base on the moon.
"""

if __name__ == "__main__":
    sample_text = """..."""
    claims = extract_claims(sample_text)

    print("Extracted Claims:")
    for i, c in enumerate(claims, 1):
        print(f"{i}. {c}")