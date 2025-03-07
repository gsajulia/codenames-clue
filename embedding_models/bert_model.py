from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import torch
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
from utils.functions import touch_cosine_similarity, touch_euclidean_distance
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

    # def find_hint(self, target_words, avoid_words, k=3): # TODO creating new function to find hint based in NN
    #     """
    #     Finds the best hint word that is most similar to each target word individually.
        
    #     Args:
    #         target_words (list of str): Words to prioritize.
    #         avoid_words (list of str): Words to avoid.
    #         k (int): Number of nearest neighbors to consider for each target word.
        
    #     Returns:
    #         dict: Mapping of each target word to its best hint and neighbors.
    #     """
        
    #     print("Computing embeddings for target words...")
    #     target_embeddings = self.get_word_embedding(target_words)
    #     target_embeddings = self.weighted_target_embedding(target_embeddings)
        
    #     print("Precomputing embeddings for candidate words...")
    #     candidate_words = [
    #         word for word in tqdm(self.tokenizer.vocab.keys(), desc="Checking words with Bert") 
    #         if self.is_valid_hint(word, target_words)
    #     ]
        
    #     candidate_embeddings = []
    #     for batch_start in tqdm(range(0, len(candidate_words), 16), desc="Processing candidate words"):
    #         batch = candidate_words[batch_start:batch_start + 16]
    #         embedding_batch = self.get_word_embedding(batch)
    #         candidate_embeddings.append(embedding_batch)
        
    #     candidate_embeddings = torch.cat(candidate_embeddings, dim=0)  # (M, D) -> M candidates

    #     results = {}

    #     for i, target_embedding in enumerate(target_embeddings):  
    #         target_embedding = target_embedding.unsqueeze(0)  # (1, D)
            
    #         # Compute cosine similarity between this specific target and candidates
    #         similarities = F.cosine_similarity(candidate_embeddings, target_embedding)

    #         # Get the top-k nearest neighbors
    #         top_k_indices = torch.topk(similarities, k=k).indices
    #         top_k_words = [candidate_words[i] for i in top_k_indices]
            
    #         # Best hint word is the first nearest neighbor
    #         best_index = top_k_indices[0].item()
    #         best_hint = candidate_words[best_index]
    #         best_score = similarities[best_index].item()

    #         results[target_words[i]] = {
    #             "best_hint": best_hint,
    #             "best_score": best_score,
    #             "top_k_neighbors": top_k_words
    #         }

    #         print(f"Target: {target_words[i]}, Best Hint: {best_hint}, Score: {best_score}, Neighbors: {top_k_words}")

    #     return results

    
    def find_hint(self, target_words, avoid_words):
        """
        Finds the best hint word that is most similar to the target words 
        and least similar to the avoid words.
        Args:
            target_words (list of str): Words to prioritize.
            avoid_words (list of str): Words to avoid.
        Returns:
            str: The best hint word.
        """
        print("Computing embeddings for target and avoid words...")
        target_embeddings = self.get_word_embedding(target_words)
        target_embeddings = self.weighted_target_embedding(target_embeddings)
        # avoid_embeddings = self.get_word_embedding(avoid_words)
        # target_embeddings = target_embeddings.mean(dim=0, keepdim=True) # TODO use this only for showing results
        
        print("Precomputing embeddings for candidate words...")
        candidate_words = [
            word for word in tqdm(self.tokenizer.vocab.keys(), desc="Checking words with Bert") 
            if self.is_valid_hint(word, target_words)
        ]
        candidate_embeddings = []
        for batch_start in tqdm(range(0, len(candidate_words), 16), desc="Processing candidate words"):
            batch = candidate_words[batch_start:batch_start + 16]
            candidate_embeddings.append(self.get_word_embedding(batch))
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
    

        target_similarities = touch_cosine_similarity(candidate_embeddings, target_embeddings)
        # target_similarities = [self.euclidean_distance(candidate_embeddings, target_embeddings) for candidate_embeddings in candidate_embeddings]
        # target_distances_tensor = torch.tensor(target_similarities)
        # best_index = scores.argmin().item()
        #avoid_similarities = self.cosine_similarity(candidate_embeddings, avoid_embeddings)
        
        # top_k_indices = torch.topk(target_similarities, k=3).indices
        # top_k_words = [candidate_words[i] for i in top_k_indices]
        
        scores = target_similarities # - avoid_similarities

        best_index = scores.argmax().item()
        best_hint = candidate_words[best_index]
        best_score = scores[best_index].item()

        print(f"Best hint: {best_hint}, Score: {best_score}")
        return {"best_hint": best_hint, "best_score":best_score}