import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Install dependencies
st.title("Installing dependencies...")
st.write("Installing required dependencies. This may take a few moments...")
st.code("pip install matplotlib")

import matplotlib.pyplot as plt  # Importing matplotlib after installation

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
user_word = st.sidebar.text_input("Enter a word to get its vector:", "")

# Main area for display output
if st.sidebar.button('Get Word Vector'):
    if user_word:
        try:
            word_vector = model.wv[user_word]  # Get the vector for the user input word
            st.write(f"Vector for '{user_word}': {word_vector}")

            # Get the next 5 similar words and their vectors
            similar_words = model.wv.most_similar(user_word, topn=5)
            st.write("Next 5 words similar to", user_word, ":", [word for word, _ in similar_words])
            
            # Plotting the word vectors
            fig, ax = plt.subplots()
            ax.bar(range(len(word_vector)), word_vector, label=user_word)  # Plot user input word vector
            for word, _ in similar_words:
                similar_word_vector = model.wv[word]
                ax.bar(range(len(similar_word_vector)), similar_word_vector, label=word, alpha=0.5)  # Plot similar word vector
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Value')
            ax.set_title(f"Word Vectors for '{user_word}' and Similar Words")
            ax.legend()
            st.pyplot(fig)
        except KeyError:
            st.error(f"Word '{user_word}' not found in the vocabulary.")
    else:
        st.warning("Please enter a word.")
