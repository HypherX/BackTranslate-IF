def evaluate_exact_word_count_per_sentence(response: str, target_words: int) -> bool:
    """
    Evaluates if all sentences in `response` contain exactly `target_words` words.
    Uses NLTK for sentence and word tokenization.

    Args:
        response (str): The text to evaluate.
        target_words (int): The exact number of words per sentence.

    Returns:
        bool: True if all sentences meet the word count requirement, False otherwise.
    """
    if not isinstance(response, str) or not isinstance(target_words, int):
        return False

    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize

    # Ensure necessary NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    sentences = sent_tokenize(response)

    for sentence in sentences:
        import pdb
        pdb.set_trace()
        words = word_tokenize(sentence)
        if len(words) != target_words:
            return False

    return True

if __name__ == "__main__":
    print(evaluate_exact_word_count_per_sentence(
        "I love you, too. Yes, I am, too.",
        4
    ))