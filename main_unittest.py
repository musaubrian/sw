import unittest
from string import punctuation
from main import Tokenizer, Embeddings, Vocab


base_text = "Kuna aina tisa za viwakilishi:"
tokenized_words = Tokenizer(base_text).by_words()
tokenized_chars = Tokenizer(base_text).by_chars()


def contains_punctuation(input_list):
    for item in input_list:
        if any(char in punctuation for char in str(item)):
            return True
        return False


class TestTokenizer(unittest.TestCase):
    expected_char_count = 29
    expected_word_count = 5

    def test_char_count(self):
        self.assertEqual(
            len(tokenized_chars),
            self.expected_char_count,
            f"Expected {self.expected_char_count}",
        )

    def test_word_count(self):
        self.assertEqual(
            len(tokenized_words),
            self.expected_word_count,
        )

    def test_punctuation_strip(self):
        self.assertFalse(contains_punctuation(tokenized_words))


class TestVocab(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocab(tokenized_words).create()

    def test_vocab_size(self):
        """Ensure vocabulary contains unique words + <UNK>"""
        expected_vocab_size = len(set(tokenized_words)) + 1  # Unique words + <UNK>
        self.assertEqual(len(self.vocab), expected_vocab_size)

    def test_vocab_contains_UNK(self):
        """Ensure <UNK> token exists in vocab"""
        self.assertIn("<UNK>", self.vocab)

    def test_vocab_assigns_unique_ids(self):
        """Ensure all words get unique IDs"""
        unique_ids = set(self.vocab.values())
        self.assertEqual(len(unique_ids), len(self.vocab))


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocab(tokenized_words).create()
        self.embeddings = Embeddings(tokenized_words, self.vocab).create()

    def test_embeddings_length(self):
        """Ensure embeddings match tokenized input length"""
        self.assertEqual(len(self.embeddings), len(tokenized_words))

    def test_embeddings_use_vocab_ids(self):
        """Ensure embeddings contain only valid vocab IDs"""
        valid_ids = set(self.vocab.values())
        for token_id in self.embeddings:
            self.assertIn(token_id, valid_ids)


if __name__ == "__main__":
    unittest.main()
