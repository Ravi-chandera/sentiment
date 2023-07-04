import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def roberta_model(sample1):
    encodedInputToModel = tokenizer(sample1, return_tensors="pt")
    output = model(**encodedInputToModel)
    scores = output[0][0].detach().numpy()
    scoresDict = {
        "Roberta_neg": scores[0],
        "Roberta_nue": scores[1],
        "Roberta_pos": scores[2]
    }
    return scoresDict

# Streamlit app
def main():
    st.title("Sentiment Analysis with Roberta Model")
    user_input = st.text_input("Enter text for sentiment analysis")
    
    if user_input:
        scores = roberta_model(user_input)
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        sentiment_probs = softmax(list(scores.values()))
        
        st.subheader("Sentiment Analysis Results:")
        for label, prob in zip(sentiment_labels, sentiment_probs):
            st.write(f"{label}: {prob:.2%}")
    
if __name__ == "__main__":
    main()
