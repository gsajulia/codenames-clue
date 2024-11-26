import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
#nltk.download('wordnet')
#nltk.download('omw-1.4')

class GloveModel:
    def __init__(self, glove_path):
        """
        Initializes the GloVe model by loading embeddings.
        """
        self.embeddings = self._load_glove_embeddings(glove_path)
        self.lemmatizer = WordNetLemmatizer()
        print("GloVe model is ready.")

    def _load_glove_embeddings(self, file_path):
        """
        Loads GloVe embeddings from the specified file.
        """
        print("Loading GloVe embeddings...")
        embeddings = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                embeddings[word] = vector
        print("Embeddings loaded!")
        return embeddings

    def cosine_similarity(self, vec1, vec2):
        """
        Computes the cosine similarity between two vectors.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def similarity_from_distance(self, vec1, vec2):
        """
        Computes the similarity between two vectors based on Euclidean distance.
        """
        # Compute Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        return 1 - (distance ** 2 / 2)

    def lemmatize_word(self, word):
        """
        Lemmatizes a word to its base form (removes derivations).
        """
        return self.lemmatizer.lemmatize(word.lower())

    def find_hint(self, target_words, avoid_words):
        """
        Finds the best hint based on the provided target and avoid word arrays.
        """
        # Lemmatize the words to avoid comparing different word forms
        target_words = [self.lemmatize_word(word) for word in target_words]
        avoid_words = [self.lemmatize_word(word) for word in avoid_words]

        # Filter out words that don't exist in embeddings
        target_vectors = [self.embeddings[word] for word in target_words if word in self.embeddings]
        avoid_vectors = [self.embeddings[word] for word in avoid_words if word in self.embeddings]

        best_hint = None
        best_score = float('-inf')

        # Iterate through the GloVe embeddings to find the best hint
        for candidate, candidate_vec in self.embeddings.items():
            if candidate not in target_words:  # Exclude target words as hints
                target_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in target_vectors) / len(target_vectors)
                avoid_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in avoid_vectors) / len(avoid_vectors)
                
                score = target_sim - avoid_sim
                if score > best_score:
                    best_hint = candidate
                    best_score = score

        return best_hint
