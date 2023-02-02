"""

	Simple Streamlit webserver application for serving developed classification
	models.

	Author: Explore Data Science Academy.

	Note:
	---------------------------------------------------------------------
	Please follow the instructions provided within the README.md file
	located within this directory for guidance on how to use this script
	correctly.
	---------------------------------------------------------------------

	Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import time
import datetime
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Data dependencies
import pandas as pd

# Create lemmatization class and define lemmatization function within it
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, message):
        return [self.wnl.lemmatize(t) for t in word_tokenize(message)]

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	st.write('<base target="_blank">', unsafe_allow_html=True)
	prev_time = [time.time()]

a, b = st.columns([1, 10])

with a:
		st.text("")
st.image("logoOfficial.png", width=50)
with b:
    st.title("Tweet Classifer")

st.write("Climate change tweet classification")

st.subheader("""Welcome to our Tweet Classifier.""")
st.write()


	# Creating sidebar with selection box -
	# you can create multiple pages this way
options = ["Prediction", "Information"]
selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
if selection == "Information":
		st.info("""We, at the Data Whispers have created this tool that allows you to input any tweet to determine its sentiments with regards to man- made climate change.
    All you need to is paste whatever tweet you want in the text box provided on the Home Page and click the classify button.
    """)
		# You can read a markdown file from supporting resources folder
		st.markdown("A lot of work has been put into creating this web app and we hope that it serves you well")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
if selection == "Prediction":

		st.info("""Sentiment Description

 2 = News: the tweet links to factual news about climate change

 1 = Pro: the tweet supports the belief of man-made climate change
 
 0 = Neutral: the tweet neither supports nor refutes the belief of man-made climate change

-1 = Anti: the tweet does not believe in man-made climate change""")


		st.info("Prediction using our classification model.")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Tweet","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/tuned_SVM.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()# s