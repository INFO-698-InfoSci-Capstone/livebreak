import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)             # Remove HTML tags
    text = re.sub(r"[\n\r\t]+", " ", text)          # Normalize whitespace characters
    text = re.sub(r"\s{2,}", " ", text)             # Collapse multiple spaces
    return text.strip()

def preprocess_with_spacy(text):
    doc = nlp(clean_text(text))
    return list(doc.sents)  # return actual sentence objects

# Test
sample_text = """
<p>NASA is sending astronauts back to the moon.</p>\n\n A total of 12 astronauts have gone to the moon. 
One famous quote is "One small step for man, one giant leap for mankind." <br><br>    
NASA is planning a base on the moon.
"""

if __name__ == "__main__":
    sample_text = """..."""
    sentences = preprocess_with_spacy(sample_text)

    print("Preprocessed Sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")