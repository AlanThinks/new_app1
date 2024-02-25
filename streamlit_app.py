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
            st.write("Next 5 words similar to", user_word, ":")
            for word, vector in similar_words:
                st.write(f"{word}: {vector}")
            
            # Prepare data for scatterplot
            word_vectors = np.vstack([word_vector] + [model.wv[word] for word, _ in similar_words])
            words = [user_word] + [word for word, _ in similar_words]
            x = word_vectors[:, 0]
            y = word_vectors[:, 1]

            # Plotting the word vectors
            fig, ax = plt.subplots()
            ax.scatter(x, y, label=user_word, color='blue')  # Scatter plot for user input word
            for i, word in enumerate(words):
                ax.annotate(word, (x[i], y[i]), textcoords="offset points", xytext=(5,5), ha='center')  # Annotate similar words with coordinates
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f"Word Vectors for '{user_word}' and Similar Words")
            ax.legend()
            st.pyplot(fig)
        except KeyError:
            st.error(f"Word '{user_word}' not found in the vocabulary.")
    else:
        st.warning("Please enter a word.")
