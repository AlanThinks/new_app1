import os
os.system('pip install matplotlib')
import streamlit as st
import pickle


# Define a function to load the model and apply the st.cache decorator
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model_cbow.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the pickled model using the cached function
model = load_model()


# Setting up the sidebar
st.sidebar.title("Options")
st.sidebar.info("This NLP app uses a pre-trained model to check word2vec on the script of American Psycho. For example type the word Bateman")

# Main application
st.title('Word2Vec')

# User input in sidebar
#user_input = st.sidebar.text_area("Enter Text for Analysis", "")

# Main area for display output
user_word = st.sidebar.text_input("Enter a word to get its vector:", "")

# Main area for display output
if st.sidebar.button('Get Word Vector'):
    if user_word:
        try:
            word_vector = model.wv[user_word]  # Get the vector for the user input word
            st.write(f"Vector for '{user_word}': {word_vector}")
        except KeyError:
            st.error(f"Word '{user_word}' not found in the vocabulary.")
    else:
        st.warning("Please enter a word.")
