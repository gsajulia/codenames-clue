from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel
from utils.generate_board import GenerateBoard
from utils.batch_results import BatchResults
from algorithms.find_closest_cards import get_clue_candidates
import pandas as pd

if __name__ == "__main__":
    board = GenerateBoard()
    
    try:
        board_cards = board.get_codenames_board()
        # default_target = ['muggle']
        default_target = ['muggle', 'foot', 'knock', 'sky', 'rice', 'sea', 'pearl', 'guitar'] # medium case / test with coef 0.9
        # default_target = ['shiny', 'sparkle', 'knock', 'sky', 'rice', 'sea', 'pearl', 'guitar'] # happy case
        default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'log', 'sea',]
        
        print("target", default_target)
        print("\n\n avoid", default_avoid)

        target_words = default_target
        avoid_words = default_avoid
        
        print('Selecting words for BERT')
        bert_model = BertModel()
        target_embeddings_bert = bert_model.get_word_embedding(target_words).cpu().numpy()
        clue_candidates_bert = get_clue_candidates(target_words, target_embeddings_bert)
        print(clue_candidates_bert)
        model_result = bert_model.find_hint(clue_candidates_bert, avoid_words)
        # print(f"Suggested Bert hint: {model_result["best_hint"]}  {model_result["best_score"]}\n")
        # model_result = bert_model.find_hint(['foot'], avoid_words)
        # print(f"Suggested Bert hint: {model_result["best_hint"]}  {model_result["best_score"]}\n")
        # model_result = bert_model.find_hint(['burn'], avoid_words)
        # print(f"Suggested Bert hint: {model_result["best_hint"]}  {model_result["best_score"]}\n")
        # model_result = bert_model.find_hint(['nobel'], avoid_words)
        # print(f"Suggested Bert hint: {model_result["best_hint"]}  {model_result["best_score"]}\n")
        # model_result = bert_model.find_hint(['pig'], avoid_words)
        # print(f"Suggested Bert hint: {model_result["best_hint"]}  {model_result["best_score"]}\n")
        # print('Selecting words for Glove')
        # glove_model = GloveModel()
        # target_embeddings_glove = glove_model.get_word_embedding(target_words)
        # clue_candidates_glove = get_clue_candidates(target_words, target_embeddings_glove)
        # print(clue_candidates_glove)
        # model_result = glove_model.find_hint(clue_candidates_glove, avoid_words)
        # print(f"Suggested Glove hint: {model_result["best_hint"]} {model_result["best_score"]}\n")        
        
        # Generate results
        # default_target = board_cards["target"]
        # default_avoid = board_cards["avoid"]
        # batch_results = BatchResults()
        # batch_results.get_batch_results()

    except Exception as e:
        print(f"An error occurred: {e}\n")
