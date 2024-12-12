import pandas as pd
import re

def clean_comment(comment):
    if not isinstance(comment, str):  # Ensure the input is a string
        return ""
    comment = re.sub(r"http\S+", "", comment)  # Remove URLs
    comment = re.sub(r"[^a-zA-Z\s]", "", comment)  # Remove special characters
    return comment.lower().strip()  # Convert to lowercase and strip whitespaces

def preprocess_data(file_path, output_path):
    df = pd.read_csv(file_path)
    df['comment'] = df['comment'].apply(clean_comment)
    df = df[df['comment'] != ""]  # Remove empty comments after cleaning
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("data/categorized_comments.csv", "data/cleaned_categorized_comments.csv")
