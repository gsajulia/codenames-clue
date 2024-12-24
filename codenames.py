from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel
from generate_board import GenerateBoard

if __name__ == "__main__":
    board = GenerateBoard()
    board_cards = board.get_codenames_board()
    default_target = board_cards["target"]
    default_avoid = board_cards["avoid"]
    
    # default_target = ['children', 'classroom']
    # default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'log', 'sea',]
    
    print("red", default_target)
    print("\n\nblue", default_avoid)

    target_words = default_target
    avoid_words = default_avoid

    try:
        model = GloveModel()
        hint = model.find_hint(target_words, avoid_words)
        print(f"Suggested Glove hint: {hint}\n")
        model = BertModel()
        hint = model.find_hint(target_words, avoid_words)
        print(f"Suggested Bert hint: {hint}\n")
    except Exception as e:
        print(f"An error occurred: {e}\n")
