import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Step 1: Load the dataset
file_path = "job_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("job_data.csv not found. Please place it in the same directory.")

df = pd.read_csv(file_path)

# Step 2: Data Cleaning (optional basic check)
df.dropna(inplace=True)  # Remove rows with missing values

# Step 3: Combine relevant features into one text column
df['combined'] = df['Skills'] + " " + df['Education'] + " " + df['Description']

# Step 4: Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

# Step 5: Fit-transform the combined column
job_vectors = tfidf.fit_transform(df['combined'])

# Step 6: Save model components to a .pkl file
with open("model.pkl", "wb") as file:
    pickle.dump((df, tfidf, job_vectors), file)

print("✅ Model training completed and saved as model.pkl.")

# Step 7: Test the model with a sample resume (optional)
sample_resume = """
I am a B.Tech Computer Science graduate. I have experience in Python, Flask, and SQL. 
I built backend APIs and deployed web apps using Django and Flask.
"""

# Step 8: Vectorize the resume
resume_vector = tfidf.transform([sample_resume])

# Step 9: Calculate cosine similarity
similarities = cosine_similarity(resume_vector, job_vectors)

# Step 10: Get top 5 matching jobs
top_indices = similarities[0].argsort()[::-1][:5]
top_jobs = df.iloc[top_indices][['Job_Title', 'Description']]

# Step 11: Print top job recommendations
print("\n📌 Top 5 job recommendations based on sample resume:")
for idx, row in top_jobs.iterrows():
    print(f"\n🔹 {row['Job_Title']}")
    print(f"   {row['Description']}")

