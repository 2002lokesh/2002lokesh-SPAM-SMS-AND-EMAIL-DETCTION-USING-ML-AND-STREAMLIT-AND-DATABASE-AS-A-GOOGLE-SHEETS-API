import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Initialize gspread credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("sample.json", scope)
client = gspread.authorize(creds)

# Initialize NLTK
nltk.download("punkt")
nltk.download("stopwords")
ps = PorterStemmer()

# Load pre-trained models
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# Use the HTML template
st.set_page_config(
    page_title="Spam Classifier",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Header
header_html = "<h1 class='text-center'>📧 Email/SMS Spam Classifier</h1>"
st.markdown(header_html, unsafe_allow_html=True)

# Sidebar
about_html = "<h4>ℹ️ About</h4><p>This app predicts whether an input message is spam or not using a pre-trained model.</p>"
st.sidebar.markdown(about_html, unsafe_allow_html=True)

# Input Text Area
input_sms = st.text_area("✉️ Enter the message")

# Predict Button
if st.button("🚀 Predict", key="predict_btn"):
    # Preprocess
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = [i for i in text if i.isalnum()]
        y = [i for i in y if i not in stopwords.words("english") and i not in string.punctuation]
        y = [ps.stem(i) for i in y]

        return " ".join(y)

    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display Result
    result_message = f"<div class='result-message'>🔍 Analysis Complete: {'Spam' if result == 1 else 'Not Spam'}!</div>"
    st.markdown(result_message, unsafe_allow_html=True)

    # Write to Google Sheets
    sheet_url = "https://docs.google.com/spreadsheets/d/12MrdL6UMnFW7GSwSmHvPUAKiejXy4Q9WUd5OpCjh390/edit?usp=sharing"
    sheet_name = "SpamPredictions"
    sheet = client.open_by_url(sheet_url).sheet1

    # Append only the necessary information to columns A and B
    sheet.append_row([input_sms, "Spam" if result == 1 else "Not Spam"], value_input_option='USER_ENTERED')

    # Resize the height of the first row to fit the content
    sheet.resize(rows=len(sheet.get_all_values()) + 1)

    # Redirect to a new page with prediction results
    new_page_content = f"""
    <h2 class='text-center'>Prediction Results</h2>
    <p class='text-center'>The input message is classified as: {'Spam' if result == 1 else 'Not Spam'}.</p>
    """
    st.markdown(new_page_content, unsafe_allow_html=True)
