from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from io import StringIO

app = Flask(__name__)

## DATA TRAINING

data = """
label, isikomen
"Bad","dua dua nya cantik, cowoknya aja yang red flag"
"Bad","terbukti kan cwek lbih suka cwok brngsek"
"Bad","Njir ada manusia modelan gini"
"Good","Manis banget yaampuun"
"Good","Rejekinya lancar terus yaa"
"Good","Produknya bagus dan cocok di aku, sukak banget "
"""

df = pd.read_csv(StringIO(data))
df.columns = df.columns.str.strip()

df['isikomen'].fillna('', inplace=True)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(df['isikomen'])
y_train = df['label']
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        comments = data.get('comments')

        comments = [comment if comment is not None else '' for comment in comments]

        comments_tfidf = vectorizer.transform(comments)

        predictions = clf.predict(comments_tfidf)

        response = {"predictions": list(predictions)}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
