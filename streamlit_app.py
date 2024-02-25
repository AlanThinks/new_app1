import os
os.system('pip install matplotlib')
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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
st.sidebar.info("This NLP app uses a pre-trained model to check word2vec on the script of American Psycho. For example, type the word Bateman")

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

            # Plot diagram
            words = [user_word]
            vectors = [model.wv[user_word]]

            # Debugging statements to inspect vectors
            st.write("Shape of vectors:", np.array(vectors).shape)
            st.write("Example vector:", vectors[0])

            # Check if the vectors have more than 1 dimension
            if len(vectors[0]) > 1:
                # Perform PCA
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(vectors)
                principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=words)

                # Plot
                fig, ax = plt.subplots()
                ax.scatter(principal_df['PC1'], principal_df['PC2'])
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.annotate(user_word, (principal_df['PC1'][0], principal_df['PC2'][0]))  # Annotate the user input word
                st.pyplot(fig)
            else:
                st.error("Word vectors have only one dimension and cannot be plotted.")
        except KeyError:
            st.error(f"Word '{user_word}' not found in the vocabulary.")
    else:
        st.warning("Please enter a word.")
