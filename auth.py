import praw

def get_reddit_instance():
    return praw.Reddit(
        client_id="UFcdSWGJMlvZ3uwJ_SgxSQ",
        client_secret="9LcyuvJE8lNJPwnpjO6_s-IGmG0XKA",
        user_agent="Political Sentiment Analysis by /u/BeneficialGap942"
    )
