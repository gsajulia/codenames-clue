import numpy as np
import keyboard

class CodenamesModel:
    def __init__(self, glove_path):
        """
        Initializes the model by loading GloVe embeddings only once.
        """
        self.embeddings = self._load_glove_embeddings(glove_path)
        print("Model is ready. Use `find_hint` with your targets and avoids.")

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

    def find_hint(self, target_words, avoid_words):
        """
        Finds the best hint based on the provided target and avoid word arrays.
        """
        target_vectors = [self.embeddings[word] for word in target_words if word in self.embeddings]
        avoid_vectors = [self.embeddings[word] for word in avoid_words if word in self.embeddings]

        best_hint = None
        best_score = float('-inf')

        for candidate, candidate_vec in self.embeddings.items():
            # Rule to avoid suggesting target words as hints
            if candidate not in target_words:
                # Similarity with target words
                target_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in target_vectors) / len(target_vectors)
                # Similarity with avoid words
                avoid_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in avoid_vectors) / len(avoid_vectors)
                
                # Score: closer to target words, further from avoid words
                score = target_sim - avoid_sim
                if score > best_score:
                    best_hint = candidate
                    best_score = score

        return best_hint

if __name__ == "__main__":
    # Initialize the model only once
    glove_path = "embeddings/glove.6b/glove.6B.300d.txt"
    model = CodenamesModel(glove_path)

    print("\nWelcome to the Codenames hint generator!")
    print("Enter your target and avoid words to get a hint.")
    print("Type 'exit' at any time to quit.\n")
    # All target 'blind', 'minotaur', 'genie', 'new', 'telescope', 'sugar', 'flower', 'puppet', 'cat', 'dwarf', 'good'
    default_target = ['minotaur', 'dwarf']
    default_avoid = ['light', 'honey', 'milk', 'bunk', 'cycle', 'orange', 'mermaid', 'sink', 'mine', 'river','cloud', 'diamond'] # assassin: diamond

    while True:
        # Input target words (your team cards)
        target_input = input("Enter target words (comma-separated) or enter to use default: ")
        if target_input.lower() == 'exit':
            print("Goodbye!")
            break
        if keyboard.read_event().name == 'enter':
            target_words = default_target
        else:
            target_words = [word.strip() for word in target_input.split(",")]

        # Input avoid words (enemy or assassin cards)
        avoid_input = input("Enter avoid words (comma-separated) or enter to use default: ")
        if avoid_input.lower() == 'exit':
            print("Goodbye!")
            break
        if keyboard.read_event().name == 'enter':
            avoid_words = default_avoid
        else:
            avoid_words = [word.strip() for word in avoid_input.split(",")]

        # Generate hint
        try:
            hint = model.find_hint(target_words, avoid_words)
            print(f"Suggested hint: {hint}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")
