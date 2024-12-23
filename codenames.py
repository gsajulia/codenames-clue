from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel

if __name__ == "__main__":
    # All target 'blind', 'minotaur', 'genie', 'new', 'telescope', 'sugar', 'flower', 'puppet', 'cat', 'dwarf', 'good'
    #default_target = ['dolphin', 'wale']
    #default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'log', 'sea', 'sun'] # assassin: diamond
    default_target = ['children', 'classroom']
    default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'mar', 'log', 'sea', 'sun']
    print("Choose the model to use:")
    print("1. GloVe")
    print("2. BERT")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        glove_path = "embeddings/glove.6b/glove.6B.300d.txt"
        model = GloveModel(glove_path)
    elif choice == "2":
        model = BertModel()
    else:
        print("Invalid choice!")
        exit()

    target_words = default_target
    avoid_words = default_avoid

    try:
        hint = model.find_hint(target_words, avoid_words)
        print(f"Suggested hint: {hint}\n")
    except Exception as e:
        print(f"An error occurred: {e}\n")
