import numpy as np


class Tokenizer:
    def __init__(self, text: str):
        self.text: str = text.lower().strip()

    def by_words(self) -> np.array:
        return np.array(self.text.split())

    def by_chars(self):
        chars = []
        for word in self.text.split():
            chars.extend(list(word))
            chars.append(" ")
        return chars


class Vocab:
    def __init__(self, tokens: np.array):
        self.tokens = tokens
        self.vocabulary: dict[str, int] = {"<UNK>": 0}

    def create(self):
        id = len(self.vocabulary)
        for token in self.tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = id
                id += 1

        return self.vocabulary


class Embeddings:
    def __init__(self, tokens: np.array, vocab: dict[str, int]):
        self.vocab = vocab
        self.tokens = tokens
        self.embeddings = np.array([])

    def create(self) -> np.array:
        for token in self.tokens:
            if token not in self.vocab:
                self.embeddings = np.append(self.embeddings, self.vocab["<UNK>"])
            else:
                self.embeddings = np.append(self.embeddings, self.vocab[token])

        return self.embeddings


text = """Aina za maneno ni dhana au maana ya neno/maneno. Pia aina za maneno huhusisha mgawanyo wa maneno hayo kulingana na matumizi yake.

Kwanza neno ni umbo lenye maana ambalo lina nafasi pande mbili. Neno ni silabi au mkusanyo wa silabi wenye kubeba au kuleta maana fulani."""

if __name__ == "__main__":
    tokens = Tokenizer(text).by_words()
    vocabulary = Vocab(tokens).create()
    test_str = "maneno ya maana"
    test_tokens = Tokenizer(test_str).by_words()
    print(vocabulary, "\n", test_tokens)
    print(Embeddings(test_tokens, vocabulary).create())
