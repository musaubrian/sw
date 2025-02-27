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
        self.embeddings = np.array([], dtype=int)

    def create(self) -> np.array:
        for token in self.tokens:
            if token not in self.vocab:
                self.embeddings = np.append(self.embeddings, self.vocab["<UNK>"])
            else:
                self.embeddings = np.append(self.embeddings, self.vocab[token])

        return self.embeddings


class Neuron:
    def __init__(self, input_dim, hidden_dim):
        self.W_in = np.random.randn(hidden_dim, input_dim)  # Input weights
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim)  # Hidden state weights
        self.b = np.zeros((hidden_dim,))

    def step(self, input_token, previous_hidden_state):
        input_contribution = np.dot(self.W_in, input_token)
        memory_contribution = np.dot(self.W_hidden, previous_hidden_state)
        new_hidden_state = np.tanh(input_contribution + memory_contribution + self.b)

        return new_hidden_state


class SRNN:
    def __init__(self):
        self.weights = np.array([])

    def forward(self):
        pass

    def predict_next(self):
        pass


text = """Aina za maneno ni dhana au maana ya neno/maneno. Pia aina za maneno huhusisha mgawanyo wa maneno hayo kulingana na matumizi yake.

Kwanza neno ni umbo lenye maana ambalo lina nafasi pande mbili. Neno ni silabi au mkusanyo wa silabi wenye kubeba au kuleta maana fulani."""

if __name__ == "__main__":
    tokens = Tokenizer(text).by_words()
    vocabulary = Vocab(tokens).create()
    test_str = "maneno ya maana"
    test_tokens = Tokenizer(test_str).by_words()
    print(vocabulary, "\n", test_tokens)
    embedded_test_tokens = Embeddings(test_tokens, vocabulary).create()
    print(embedded_test_tokens)

    input_dim = len(vocabulary)
    hidden_dim = 3
    neuron = Neuron(input_dim, hidden_dim)
    hidden_state = np.zeros((hidden_dim,))

    for token_embedding in embedded_test_tokens:
        # Convert token index to a one-hot vector
        input_vector = np.zeros((input_dim,))
        if int(token_embedding) != -1:
            input_vector[token_embedding] = 1

        # Pass through RNN step function
        hidden_state = neuron.step(input_vector, hidden_state)
        print("New Hidden State:", hidden_state)
