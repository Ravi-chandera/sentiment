from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    
    if user_input:
        scores = roberta_model(user_input)
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        sentiment_probs = softmax(list(scores.values()))
        
        return render_template('results.html', labels=sentiment_labels, probs=sentiment_probs)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
