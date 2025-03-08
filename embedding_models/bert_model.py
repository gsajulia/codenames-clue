from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import torch
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
from utils.functions import torch_cosine_similarity, torch_euclidean_distance
from sklearn.neighbors import NearestNeighbors

nltk.download('stopwords')
nltk.download('wordnet')

class BertModel:
    def __init__(self, model_name="bert-base-uncased"):
        """
        Initializes the BERT model and tokenizer.
        Args:
            model_name (str): Name of the pretrained BERT model to load.
        """
        print("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        print("BERT model loaded!")

    def lemmatize_word(self, word):
        """
        Lemmatizes a word to its base form (removes derivations).
        """
        return self.lemmatizer.lemmatize(word.lower())

    def is_valid_hint(self, word, target_words):
        """
        Determines if a word is valid for use as a hint.
        - Excludes stopwords and words with specific unwanted characteristics.
        """
        stop_words = set(stopwords.words('english'))
        if word.startswith("##"):
            return False
        if word in stop_words:
            return False
        if len(word) < 3:
            return False
        if word in target_words:
            return False
        # Can be used to avoid plural words
        # singular = self.stemmer.stem(word)
        # if singular != word:
        #     return False
        
        # Ensure the word is not a derivation of any target word
        lemmatized_word = self.lemmatize_word(word)
        for target_word in target_words:
            if lemmatized_word == self.lemmatize_word(target_word):
                return False

        technical_words = {}
        if any(substring in word for substring in technical_words):
            return False
        return word.isalpha()
    
    def get_word_embedding(self, words):
        """
        Computes the embedding vector for a given word using BERT.
        Args:
            word (str): The word for which the embedding is required.
        Returns:
            torch.Tensor: The embedding vector.
        """
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
    def weighted_target_embedding(self, target_embeddings):
        scores = target_embeddings.norm(dim=1)
        weights = F.softmax(scores, dim=0)
        weighted_embedding = torch.sum(target_embeddings * weights.unsqueeze(1), dim=0, keepdim=True)
        return weighted_embedding

    def process_candidates(self, target_words):
        print("Precomputing embeddings for candidate words...")
        candidate_words = [
            word for word in tqdm(self.tokenizer.vocab.keys(), desc="Checking words with Bert") 
            if self.is_valid_hint(word, target_words)
        ]
        
        candidate_embeddings = []
        for batch_start in tqdm(range(0, len(candidate_words), 16), desc="Processing candidate words"):
            batch = candidate_words[batch_start:batch_start + 16]
            embedding_batch = self.get_word_embedding(batch)
            candidate_embeddings.append(embedding_batch)
        
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
        
        return candidate_embeddings, candidate_words
        
    def select_best_hint_from_embeddings_and_neighbors(self, target_words, k=1000):
        """
        Select the best hint word that is most similar to each target word individually.
        
        Args:
            target_words (list of str): Words to prioritize.
            k (int): Number of nearest neighbors to consider for each target word.
        
        Returns:
            dict: Mapping of each target word to its best hint and neighbors.
        """
        
        print("Computing embeddings for target words...")
        embeddings = []
        for word in target_words:
            target_embeddings = self.get_word_embedding([word])
            target_embeddings = self.weighted_target_embedding(target_embeddings)
            embeddings.append(target_embeddings)
            # target_embeddings = target_embeddings.mean(dim=0, keepdim=True)
        
        candidate_embeddings, candidate_words = self.process_candidates(target_words)

        results = {}
        all_top_k_words = []
        
        for i, target_embedding in enumerate(embeddings):
            # Ensure target_embedding is 2D  
            if target_embedding.dim() == 1:
                target_embedding = target_embedding.unsqueeze(0)
            
            # Compute cosine similarity between this specific target and candidates
            similarities = torch_cosine_similarity(candidate_embeddings, target_embedding)

            # Get the top-k nearest neighbors
            top_k = torch.topk(similarities, k=k)
            top_k_indices = top_k.indices
            top_k_similarities = top_k.values
            top_k_neighbors = [(candidate_words[idx.item()], sim.item()) for idx, sim in zip(top_k_indices, top_k_similarities)]
            [all_top_k_words.append(candidate_words[i]) for i in top_k_indices]
            
    
            # Best hint word is the first nearest neighbor
            best_index = top_k_indices[0].item()
            best_hint = candidate_words[best_index]
            best_score = similarities[best_index].item()
    
            results[target_words[i]] = {
                "best_hint": best_hint,
                "best_score": best_score,
                "top_k_neighbors": top_k_neighbors
            }

        print(f"Target: {target_words[i]}, Best Hint: {best_hint}, Score: {best_score}, Neighbors: {top_k_neighbors}")

        # Now, to find the best pair (targets, neighbor) with the highest similarity:
        best_hint = None
        best_score = -1

        all_target_embedding = self.get_word_embedding(target_words)
        all_target_embedding = self.weighted_target_embedding(all_target_embedding)
        for word in all_top_k_words:  # Iterate through the lists of top-k words
            similarity = torch_cosine_similarity(self.get_word_embedding([word]), all_target_embedding)

            if similarity > best_score:
                best_score = similarity
                best_hint = word

        print(f"Best hint: {best_hint} with score {best_score}")

        return {"best_hint": best_hint, "best_score":best_score}

    
    def select_best_hint_from_embeddings(self, target_words):
        """
        Select the best hint using similarity between the embeddings
        Args:
            target_words (list of str): Team Cards
        Returns:
            str: Hint
        """
        print("Computing embeddings for target and avoid words...")
        target_embeddings = self.get_word_embedding(target_words)
        target_embeddings = self.weighted_target_embedding(target_embeddings)
        # target_embeddings = target_embeddings.mean(dim=0, keepdim=True) # TODO use this only for showing results
        
        candidate_embeddings, candidate_words = self.process_candidates(target_words)

        target_similarities = torch_cosine_similarity(candidate_embeddings, target_embeddings)
        # target_similarities = [self.euclidean_distance(candidate_embeddings, target_embeddings) for candidate_embeddings in candidate_embeddings]
        # target_distances_tensor = torch.tensor(target_similarities)
        # best_index = scores.argmin().item()
        #avoid_similarities = self.cosine_similarity(candidate_embeddings, avoid_embeddings)
        
        scores = target_similarities

        best_index = scores.argmax().item()
        best_hint = candidate_words[best_index]
        best_score = scores[best_index].item()

        print(f"Best hint: {best_hint}, Score: {best_score}")
        return {"best_hint": best_hint, "best_score":best_score}