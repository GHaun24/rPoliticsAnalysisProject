from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment_with_categories(file_path, output_path):
    analyzer = SentimentIntensityAnalyzer()
    df = pd.read_csv(file_path)

    # Ensure comments are strings and drop invalid rows
    df['comment'] = df['comment'].astype(str)  # Convert all comments to strings
    df = df[df['comment'].notna()]

    # Debugging Step: Print data types and invalid rows
    #print("Data types before processing:")
    #print(df.dtypes)

    #print("Rows with invalid or missing comments:")
    #print(df[df['comment'].isna()])

    # Apply sentiment analysis
    def categorize_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    analyze_sentiment_with_categories("data/cleaned_categorized_comments.csv", "data/sentiment_categorized_comments.csv")
