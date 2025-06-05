from flask import Flask, request, render_template
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model
df, tfidf, job_vectors = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume = request.form['resume']
        resume_vec = tfidf.transform([resume])
        scores = cosine_similarity(resume_vec, job_vectors)
        top_index = scores[0].argsort()[::-1][:3]

        recommendations = df.iloc[top_index][['Job_Title', 'Description']].to_dict(orient='records')
        return render_template('index.html', recommendations=recommendations)

    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
