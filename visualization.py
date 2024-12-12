import nltk
import ssl
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Fix SSL issue for downloading NLTK data
# This block ensures that NLTK can download the necessary stopwords dataset even if SSL certificates are not correctly configured.
try:
    _create_default_https_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords')  # Download the stopwords dataset used for filtering common words.
finally:
    ssl._create_default_https_context = _create_default_https_context

# Function to generate a pie chart for a specific category's sentiment distribution
def generate_category_pie_chart(data, title):
    sentiment_counts = data['sentiment_category'].value_counts()  # Count each sentiment category.
    plt.figure()
    plt.pie(
        sentiment_counts,  # Data for the pie chart.
        labels=sentiment_counts.index,  # Labels for each sentiment category.
        autopct='%1.1f%%',  # Display percentages on the chart.
        startangle=90,  # Start the pie chart from a 90-degree angle.
        colors=["#ff9999", "#66b3ff", "#99ff99"]  # Custom colors for better readability.
    )
    plt.title(title)  # Set the title of the pie chart.

# Function to generate a bar chart showing the most common words in a category
def generate_word_frequency_bar_chart(data, title):
    stop_words = set(stopwords.words('english'))  # Define stopwords for filtering.
    all_words = ' '.join(data['comment']).split()  # Combine all comments and split them into individual words.
    filtered_words = [word for word in all_words if word not in stop_words]  # Remove stopwords.
    word_counts = Counter(filtered_words).most_common(10)  # Get the 10 most common words.

    if len(word_counts) == 0:  # If no words are found, print a message and exit.
        print(f"No words found for {title}.")
        return

    words, counts = zip(*word_counts)  # Separate the words and their counts for plotting.
    plt.figure()
    plt.bar(words, counts, color="#6699CC")  # Create a bar chart with custom color.
    plt.title(title)
    plt.xlabel('Words')  # Label the x-axis.
    plt.ylabel('Counts')  # Label the y-axis.
    plt.xticks(rotation=45)  # Rotate the x-axis labels for readability.

# Function to generate a histogram of sentiment scores
def generate_sentiment_histogram(data, title):
    plt.figure()
    plt.hist(data['sentiment'], bins=20, color="#FFCC00", edgecolor="black")  # Create a histogram with 20 bins.
    plt.title(title)
    plt.xlabel("Sentiment Score")  # Label the x-axis.
    plt.ylabel("Frequency")  # Label the y-axis.

# Function to generate a stacked bar chart showing sentiment distribution by category
def generate_stacked_bar_chart(data, title):
    plt.figure(figsize=(10, 6))  # Set the figure size for the chart.
    sentiment_counts = data.groupby(['category', 'sentiment_category']).size().unstack(fill_value=0)  # Group data by category and sentiment.

    sentiment_counts.plot(
        kind='bar',
        stacked=True,
        color=["#ff9999", "#66b3ff", "#99ff99"],  # Define colors for each sentiment category.
        figsize=(10, 6),
        width=0.8  # Set the width of each bar.
    )

    plt.title(title)
    plt.xlabel('Category')  # Label the x-axis.
    plt.ylabel('Frequency')  # Label the y-axis.
    plt.xticks(rotation=30, ha='right')  # Rotate the x-axis labels for better visibility.
    plt.legend(title='Sentiment', loc='upper left')  # Add a legend for the chart.

# Function to generate a stacked bar chart showing sentiment distribution by subreddit
def generate_subreddit_sentiment_chart(data, title):
    plt.figure(figsize=(10, 6))
    subreddit_counts = data.groupby(['subreddit', 'sentiment_category']).size().unstack(fill_value=0)  # Group by subreddit and sentiment.

    subreddit_counts.plot(
        kind='bar',
        stacked=True,
        color=["#ff9999", "#66b3ff", "#99ff99"],
        figsize=(10, 6),
        width=0.8
    )

    plt.title(title)
    plt.xlabel('Subreddit')
    plt.ylabel('Frequency')
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Sentiment', loc='upper left')

# Function to generate a grouped bar chart comparing sentiments for Democrats and Republicans
def generate_comparison_chart(data, title):
    plt.figure(figsize=(10, 6))
    categories = ['Democrat', 'Republican']  # Define the two categories to compare.
    sentiment_labels = ['Positive', 'Neutral', 'Negative']  # Define the sentiment categories.

    # Calculate sentiment counts for each category.
    sentiment_counts = {
        category: [
            len(data[(data['category'] == category) & (data['sentiment_category'] == sentiment)])
            for sentiment in sentiment_labels
        ]
        for category in categories
    }

    x = range(len(categories))  # Define the x-axis positions for the bars.
    width = 0.2  # Set the width of each bar.

    # Create grouped bars for each sentiment category.
    for i, sentiment in enumerate(sentiment_labels):
        plt.bar(
            [pos + i * width for pos in x],
            [sentiment_counts[category][i] for category in categories],
            width=width,
            label=sentiment
        )

    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.xticks([pos + width for pos in x], categories)  # Center the x-axis labels.
    plt.legend(title='Sentiment')  # Add a legend for the grouped bars.

# Function to save all visualizations to a single PDF file
def save_visualizations_to_pdf(file_path, sentiment_data_path):
    df = pd.read_csv(sentiment_data_path)  # Read the sentiment data from the specified file.

    # Separate data for Democrats and Republicans for individual analysis.
    democrat_data = df[df['category'] == 'Democrat']
    republican_data = df[df['category'] == 'Republican']

    with PdfPages(file_path) as pdf:  # Open the PDF file to save visualizations.
        # Generate and save pie charts
        generate_category_pie_chart(democrat_data, 'Democrat Sentiment Analysis Breakdown')
        pdf.savefig()
        plt.close()

        generate_category_pie_chart(republican_data, 'Republican Sentiment Analysis Breakdown')
        pdf.savefig()
        plt.close()

        # Generate and save bar charts
        generate_word_frequency_bar_chart(democrat_data, 'Most Common Words - Democrats')
        pdf.savefig()
        plt.close()

        generate_word_frequency_bar_chart(republican_data, 'Most Common Words - Republicans')
        pdf.savefig()
        plt.close()

        # Generate and save histogram
        generate_sentiment_histogram(df, 'Overall Sentiment Distribution')
        pdf.savefig()
        plt.close()

        # Generate and save stacked bar charts
        generate_stacked_bar_chart(df, 'Sentiment Distribution by Category')
        pdf.savefig()
        plt.close()

        generate_subreddit_sentiment_chart(df, 'Sentiment Distribution by Subreddit')
        pdf.savefig()
        plt.close()

        # Generate and save comparison chart
        generate_comparison_chart(df, 'Sentiment Comparison by Category')
        pdf.savefig()
        plt.close()

        print(f"All visualizations saved to {file_path}.")

# Main block to execute the program
if __name__ == "__main__":
    save_visualizations_to_pdf(
        "results/visualizations.pdf",  # Path to save the PDF file.
        "data/sentiment_categorized_comments.csv"  # Path to the sentiment data file.
    )
