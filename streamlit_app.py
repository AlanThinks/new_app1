import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define a function to load the model and apply the st.cache decorator
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model_cbow.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the pickled model using the cached function
model = load_model()

# Main application title
st.title('Word2Vec')

# User input for word or sentence
user_input = st.text_input("Enter a word or sentence to get its vector:", "")

# Main area for display output
if user_input:
    try:
        if len(user_input.split()) == 1:  # Single word
            word = user_input
            similar_words = model.wv.most_similar(word, topn=5)
            words = [similar_word[0] for similar_word in similar_words]
            st.write(f"Next 5 words similar to '{word}':")
            for similar_word in similar_words:
                st.write(f"'{similar_word[0]}': {model.wv[similar_word[0]]}")
        else:  # Sentence
            words = user_input.split()

        word_vectors = []
        for word in words:
            word_vector = model.wv[word]  # Get the vector for each word in the input
            word_vectors.append(word_vector)
            st.write(f"Vector for '{word}': {word_vector}")

        # Prepare data for scatterplot
        word_vectors = np.array(word_vectors)
        x = word_vectors[:, 0]
        y = word_vectors[:, 1]

        # Plotting the word vectors
        fig, ax = plt.subplots()
        ax.scatter(x, y, label=user_input, color='blue')  # Scatter plot for user input word/sentence
        for i, word in enumerate(words):
            ax.annotate(f"{word}\n{x[i]:.2f}, {y[i]:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(5,5), ha='center')  # Annotate words with coordinates
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f"Word Vectors for '{user_input}'")
        ax.legend()

        # Adjusting the limits of x and y axes to ensure all points stay inside the plot
        margin = 0.1  # Add a small margin
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_ylim(min(y) - margin, max(y) + margin)

        st.pyplot(fig)
    except KeyError:
        st.error("One or more words not found in the vocabulary.")
