from bert_model_mac import BertModelMac

if __name__ == "__main__":
    # All target 'blind', 'minotaur', 'genie', 'new', 'telescope', 'sugar', 'flower', 'puppet', 'cat', 'dwarf', 'good'
    default_target = ['dolphin', 'wale']
    default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'log', 'sea', 'sun'] # assassin: diamond

    model = BertModelMac()

    print("\nWelcome to the Codenames hint generator!")
    print("Enter your target and avoid words to get a hint.")
    print("Type 'exit' at any time to quit.\n")

    target_words = default_target
    avoid_words = default_avoid

    try:
        hint = model.find_hint(target_words, avoid_words)
        print(f"Suggested hint: {hint}\n")
    except Exception as e:
        print(f"An error occurred: {e}\n")
