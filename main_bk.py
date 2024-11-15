import gradio as gr
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import data

# Sample dataset of past hires
# data = {
#     "candidate_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
#     "skills": [
#         "Python, Data Science, Machine Learning",
#         "Java, Spring, Backend Development",
#         "JavaScript, React, Frontend Development",
#         "Python, Machine Learning, Artificial Intelligence",
#         "Project Management, Agile, Leadership"
#     ],
#     "role": ["Data Scientist", "Backend Developer", "Frontend Developer", "AI Engineer", "Project Manager"]
# }


# Convert to DataFrame
past_hires_df = pd.DataFrame(data)

# TF-IDF vectorizer to convert text data into vectors for similarity comparison
vectorizer = TfidfVectorizer()


def find_matching_roles(name, skills):
    # Adding the new candidate to the data temporarily for similarity calculation
    # new_candidate = pd.DataFrame({"candidate_name": [name], "skills": [skills], "role": ["N/A"]})

    new_candidate = pd.DataFrame({"skills": [skills], "role": ["N/A"]})
    combined_data = pd.concat([past_hires_df, new_candidate], ignore_index=True)

    print("\n combined data: ", combined_data)

    # Vectorizing the skills column
    tfidf_matrix = vectorizer.fit_transform(combined_data['skills'])

    print("\n tfidf matrix: ", tfidf_matrix)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    print("\nsimilarity matrix: ", similarity_matrix)

    # Get similarity scores for the new candidate (last row in the similarity matrix)
    similarity_scores = similarity_matrix[-1][:-1]


    # Find the best matching previous hires based on similarity scores
    best_matches_idx = similarity_scores.argsort()[::-1][:3]
    best_matches = past_hires_df.iloc[best_matches_idx]

    # Format the result for display
    results = []
    for i, row in best_matches.iterrows():
        # results.append({
        #     "Similar Candidate": row['candidate_name'],
        #     "Skills": row['skills'],
        #     "Suggested Role": row['role'],
        #     "Similarity Score": round(similarity_scores[i], 2)
        # })
        results.append({
            "Skills": row['skills'],
            "Suggested Role": row['role'],
            "Similarity Score": round(similarity_scores[i], 2)
        })

    return results


# Define Gradio interface function
def gradio_interface(name, skills):
    matches = find_matching_roles(name, skills)
    # Format output for Gradio
    # output_text = "\n".join(
    #     [f"Candidate: {m['Similar Candidate']}, Role: {m['Suggested Role']}, Score: {m['Similarity Score']}" for m in
    #      matches])
    output_text = "\n".join(
        [f"Role: {m['Suggested Role']}, Score: {m['Similarity Score']}" for m in
         matches])
    return output_text


# Gradio UI with updated syntax
inputs = [gr.Textbox(label="Candidate Name"), gr.Textbox(label="Skills (comma-separated)")]
outputs = gr.Textbox(label="Suggested Roles and Similarity Scores")

gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs,
             title="Job Matching and Candidate Screening with CBR").launch()
