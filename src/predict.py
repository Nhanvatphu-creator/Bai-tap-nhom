import pickle
import os
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# load model
with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)

# load vectorizer
with open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

def predict_sentiment(text):
    clean = clean_text(text)
    text_vec = vectorizer.transform([clean])
    pred = model.predict(text_vec)[0]

NEGATIVE_KEYWORDS = ["xấu", "tệ", "thất vọng", "ồn", "móp", "hỏng", "giả", "chán", "kém"]
POSITIVE_KEYWORDS = ["tốt", "đẹp", "ngon", "hài lòng", "uy tín", "nhanh", "rẻ", "ổn", "tuyệt"]

def predict_sentiment(review: str) -> str:
    text = review.lower()
    if any(word in text for word in NEGATIVE_KEYWORDS):
        return "😡 Negative"
    if any(word in text for word in POSITIVE_KEYWORDS):
        return "😊 Positive"
    return "😐 Neutral"

    if pred == "positive":
        return "😊 Positive"
    elif pred == "negative":
        return "😡 Negative"
    else:
        return "😐 Neutral"

if __name__ == "__main__":
    review = input("Nhập review: ")
    result = predict_sentiment(review)
    print("Kết quả:", result)
    
