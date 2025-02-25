from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel
from utils.generate_board import GenerateBoard
from utils.batch_results import BatchResults
import pandas as pd

if __name__ == "__main__":
    board = GenerateBoard()
    
    try:
        board_cards = board.get_codenames_board()
        default_target = ['Muggle']
        default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'log', 'sea',]

        print("target", default_target)
        print("\n\n avoid", default_avoid)

        target_words = default_target
        avoid_words = default_avoid
        model = GloveModel()
        model_result = model.find_hint(target_words, avoid_words)
        print(f"Suggested Glove hint: {model_result["best_hint"]}\n")
        model = BertModel()
        model_result = model.find_hint(target_words, avoid_words)
        print(f"Suggested Bert hint: {model_result["best_hint"]}\n")
        
        # Generate results
        # default_target = board_cards["target"]
        # default_avoid = board_cards["avoid"]
        # batch_results = BatchResults()
        # batch_results.get_batch_results()

    except Exception as e:
        print(f"An error occurred: {e}\n")
