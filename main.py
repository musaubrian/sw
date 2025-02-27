class Tokenizer:
    def __init__(self, text: str):
        self.text: str = text.lower().strip()

    def by_words(self) -> list:
        return self.text.split()

    def by_char(self):
        chars = []
        for word in self.text.split():
            chars.extend(list(word))
        return chars


class Vocab:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.vocabulary: dict[str, int] = {"<UNK>": -1}

    def create(self):
        id = 1
        for token in self.tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = id
                id += 1

        return self.vocabulary


class Embeddings:
    def __init__(self, tokens: list[str], vocab: dict[str, int]):
        self.vocab = vocab
        self.tokens = tokens
        self.embeddings = []

    def create(self):
        for token in self.tokens:
            if token not in self.vocab:
                self.embeddings.append(-1)
            else:
                self.embeddings.append(self.vocab[token])

        return self.embeddings


text = """Aina za maneno ni dhana au maana ya neno/maneno. Pia aina za maneno huhusisha mgawanyo wa maneno hayo kulingana na matumizi yake.

Kwanza neno ni umbo lenye maana ambalo lina nafasi pande mbili. Neno ni silabi au mkusanyo wa silabi wenye kubeba au kuleta maana fulani."""

if __name__ == "__main__":
    tokens = Tokenizer(text).by_words()
    vocabulary = Vocab(tokens).create()
    print(vocabulary)
    test = "umbo ambalo lina nafasi mbili"
    test_tokens = Tokenizer(test).by_words()
    embeddings = Embeddings(test_tokens, vocabulary).create()
    print(embeddings)
