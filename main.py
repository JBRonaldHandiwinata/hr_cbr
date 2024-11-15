import gradio as gr
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import data


past_hires_df = pd.DataFrame(data)
# TF-IDF vectorizer to convert text data into vectors for similarity comparison
vectorizer = TfidfVectorizer()


def find_matching_roles(name, skills):
    new_candidate = pd.DataFrame({"skills": [skills], "role": ["N/A"]})
    combined_data = pd.concat([past_hires_df, new_candidate], ignore_index=True)

    # Vectorizing the skills column
    tfidf_matrix = vectorizer.fit_transform(combined_data['skills'])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_scores = similarity_matrix[-1][:-1]
    best_matches_idx = similarity_scores.argsort()[::-1][:3]
    best_matches = past_hires_df.iloc[best_matches_idx]

    results = []
    for i, row in best_matches.iterrows():
        results.append({
            "Skills": row['skills'],
            "Suggested Role": row['role'],
            "Similarity Score": round(similarity_scores[i], 2)
        })

    return results


def gradio_interface(name, skills):
    matches = find_matching_roles(name, skills)
    output_text = "\n".join(
        [f"Role: {m['Suggested Role']}, Score: {m['Similarity Score']}" for m in
         matches])
    return output_text


# Gradio UI with updated syntax
inputs = [gr.Textbox(label="Candidate Name"), gr.Textbox(label="Skills (comma-separated)")]
outputs = gr.Textbox(label="Suggested Roles and Similarity Scores")

gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs,
             title="Job Matching and Candidate Screening with CBR").launch()
