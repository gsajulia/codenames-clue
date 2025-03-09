from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel
from utils.generate_board import GenerateBoard
from utils.batch_results import BatchResults
import pandas as pd

if __name__ == "__main__":
    board = GenerateBoard()
    
    try:
        board_cards = board.get_codenames_board()
        # default_target = ['planets', 'jupiter']
        default_target = ['muggle']
        # default_target = ['muggle', 'foot', 'knock', 'sky', 'rice', 'sea', 'pearl', 'guitar'] # medium case / test with coef 0.9
        # default_target = ['shiny', 'foot', 'knock', 'sky', 'rice', 'sea', 'pearl', 'guitar'] # happy case
        
        print("target", default_target)
        target_words = default_target
        
        print('Selecting words for BERT')
        bert_model = BertModel()
        model_result = bert_model.select_best_hint_from_embeddings_and_neighbors(target_words)
        # model_result = bert_model.select_best_hint_from_embeddings(target_words)
        print(f"Suggested Bert hint: {model_result["best_hint"]}  {model_result["best_score"]}\n")    
        
        print('Selecting words for Glove')
        glove_model = GloveModel()
        # model_result = glove_model.select_best_hint_from_embeddings(target_words)
        model_result = glove_model.select_best_hint_from_embeddings_and_neighbors(target_words)
        print(f"Suggested Glove hint: {model_result["best_hint"]} {model_result["best_score"]}\n")   
        
        # Generate results
        # default_target = board_cards["target"]
        # default_avoid = board_cards["avoid"]
        # batch_results = BatchResults()
        # batch_results.get_fixed_batch_results()

    except Exception as e:
        print(f"An error occurred: {e}\n")
