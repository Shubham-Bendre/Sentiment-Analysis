# To run the Streamlit app, save this script as `ai2.py` and run `streamlit run Sentiment.py`

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import nltk
from nltk.corpus import movie_reviews
import streamlit as st

# Download NLTK data if not already present
nltk.download('movie_reviews')

# Load movie reviews dataset
def load_movie_reviews():
    documents = [(movie_reviews.raw(fileid), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    return documents

# Load and prepare the data
documents = load_movie_reviews()
df = pd.DataFrame(documents, columns=['text', 'label'])
df = df.sample(frac=1).reset_index(drop=True)
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with TF-IDF and Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # Use TF-IDF for better feature extraction
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print accuracy and classification report
def print_classification_report():
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
    print('Classification Report:')
    print(metrics.classification_report(y_test, y_pred))

    # Plot the classification report with only precision
    def plot_classification_report(report):
        report_df = pd.DataFrame({
            'precision': report['neg']['precision'],
            'positive': report['pos']['precision']
        }, index=['neg', 'pos'])

        plt.figure(figsize=(10, 7))
        sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Classification Report - Precision Only')
        plt.savefig('classification_report.png')  # Save the plot to a file
        print("Report:", report)

    plot_classification_report(report)

print_classification_report()

# Streamlit web app
def run_streamlit_app():
    st.title('Sentiment Analysis Web App')
    st.write("This app uses a machine learning model to analyze the sentiment of text inputs.")

    st.sidebar.title("Options")
    st.sidebar.write("You can enter text below to get sentiment predictions or view example texts.")

    user_input = st.text_area("Enter a text for sentiment analysis:", "")
    
    if user_input:
        prediction = pipeline.predict([user_input])[0]
        st.write(f'**The sentiment of the text is:** {prediction}')
        
        # Display sentiment distribution of example texts
        st.write("### Sentiment Distribution of Example Texts")
        example_texts = [
            "I love this movie! It's fantastic.",
            "This film was terrible. I hated it."
        ]
        example_predictions = [pipeline.predict([text])[0] for text in example_texts]
        example_df = pd.DataFrame({
            'Text': example_texts,
            'Predicted Sentiment': example_predictions
        })
        st.write(example_df)

        # Display a sample of the classification report image
        st.image('classification_report.png', caption='Classification Report')

run_streamlit_app()

