import numpy as np

class GloveModel:
    def __init__(self, glove_path):
        """
        Initializes the GloVe model by loading embeddings.
        """
        self.embeddings = self._load_glove_embeddings(glove_path)
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
    
    def similarity_from_distance(vec1, vec2):
        """
        Computes the similarity between two vectors based on Euclidean distance.
        """
        # Compute Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        return 1 - (distance ** 2 / 2)

    def find_hint(self, target_words, avoid_words):
        """
        Finds the best hint based on the provided target and avoid word arrays.
        """
        target_vectors = [self.embeddings[word] for word in target_words if word in self.embeddings]
        avoid_vectors = [self.embeddings[word] for word in avoid_words if word in self.embeddings]

        best_hint = None
        best_score = float('-inf')

        for candidate, candidate_vec in self.embeddings.items():
            if candidate not in target_words:  # Exclude target words as hints
                target_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in target_vectors) / len(target_vectors)
                avoid_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in avoid_vectors) / len(avoid_vectors)
                
                score = target_sim - avoid_sim
                if score > best_score:
                    best_hint = candidate
                    best_score = score

        return best_hint
