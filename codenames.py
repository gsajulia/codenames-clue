import keyboard
from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel

if __name__ == "__main__":
    # All target 'blind', 'minotaur', 'genie', 'new', 'telescope', 'sugar', 'flower', 'puppet', 'cat', 'dwarf', 'good'
    default_target = ['minotaur', 'dwarf']
    default_avoid = ['light', 'honey', 'milk', 'bunk', 'cycle', 'orange', 'mermaid', 'sink', 'mine', 'river','cloud', 'diamond'] # assassin: diamond

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

    print("\nWelcome to the Codenames hint generator!")
    print("Enter your target and avoid words to get a hint.")
    print("Type 'exit' at any time to quit.\n")

    while True:
        target_input = input("Enter target words (comma-separated): ")
        if target_input.lower() == "exit":
            print("Goodbye!")
            break
        if keyboard.read_event().name == 'enter':
            target_words = default_target
        else:
            target_words = [word.strip() for word in target_input.split(",")]

        avoid_input = input("Enter avoid words (comma-separated): ")
        if avoid_input.lower() == "exit":
            print("Goodbye!")
            break

        if keyboard.read_event().name == 'enter':
            avoid_words = default_avoid
        else:
            avoid_words = [word.strip() for word in avoid_input.split(",")]

        try:
            hint = model.find_hint(target_words, avoid_words)
            print(f"Suggested hint: {hint}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")
