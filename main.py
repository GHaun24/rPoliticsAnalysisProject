import os
from auth import get_reddit_instance
from data_collection import collect_and_categorize_comments
from data_preprocessing import preprocess_data
from sentiment_analysis import analyze_sentiment_with_categories
from visualization import save_visualizations_to_pdf

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Step 1: Authenticate with Reddit API
    print("Authenticating with Reddit API...")
    reddit = get_reddit_instance()

    # Step 2: Collect Data
    print("Collecting comments from subreddits...")
    subreddits = ["politics", "conservative", "liberal", "news", "worldnews"]
    data = collect_and_categorize_comments(reddit, subreddits, "Trump", num_threads=5, num_comments=100)
    raw_data_path = "data/categorized_comments.csv"
    data.to_csv(raw_data_path, index=False)

    # Step 3: Preprocess Data
    print("Cleaning and preprocessing data...")
    cleaned_data_path = "data/cleaned_categorized_comments.csv"
    preprocess_data(raw_data_path, cleaned_data_path)

    # Step 4: Analyze Sentiments
    print("Performing sentiment analysis...")
    sentiment_data_path = "data/sentiment_categorized_comments.csv"
    analyze_sentiment_with_categories(cleaned_data_path, sentiment_data_path)

    # Step 5: Generate Visualizations
    print("Saving visualizations to PDF...")
    save_visualizations_to_pdf("results/visualizations.pdf", sentiment_data_path)

    print("All steps completed successfully! Check the 'data/' and 'results/' folders for outputs.")

if __name__ == "__main__":
    main()
