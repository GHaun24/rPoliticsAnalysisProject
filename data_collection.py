from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def categorize_comment(comment):
    # Define keywords and phrases
    democrat_phrases = [
        "Kamala Harris", "Joe Biden", "liberal agenda", "progressive policies",
        "Democrat values", "left-wing"
    ]
    republican_phrases = [
        "Donald Trump", "MAGA", "conservative values", "GOP agenda",
        "Republican ideals", "right-wing"
    ]

    # Perform sentiment analysis
    sentiment_score = analyzer.polarity_scores(comment)['compound']

    # Categorize based on sentiment and keywords/phrases
    if any(phrase.lower() in comment.lower() for phrase in republican_phrases):
        if sentiment_score > 0.05:
            return "Republican"
        elif sentiment_score < -0.05:
            return "Democrat"
        else:
            return "Neutral-Republican"
    elif any(phrase.lower() in comment.lower() for phrase in democrat_phrases):
        if sentiment_score > 0.05:
            return "Democrat"
        elif sentiment_score < -0.05:
            return "Republican"
        else:
            return "Neutral-Democrat"

    return "Neutral"

def collect_and_categorize_comments(reddit, subreddits, keyword, num_threads=5, num_comments=100):
    comments = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Collecting data from r/{subreddit_name}...")
        for submission in subreddit.search(keyword, limit=num_threads):
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:num_comments]:
                category = categorize_comment(comment.body)
                comments.append({"subreddit": subreddit_name, "comment": comment.body, "category": category})

    return pd.DataFrame(comments)

if __name__ == "__main__":
    from auth import get_reddit_instance

    reddit = get_reddit_instance()
    subreddits = ["politics", "conservative", "liberal", "news", "worldnews"]
    data = collect_and_categorize_comments(reddit, subreddits, "Trump", num_threads=5, num_comments=100)
    data.to_csv("data/categorized_comments.csv", index=False)
