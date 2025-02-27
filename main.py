import numpy as np
import re


class Tokenizer:
    def __init__(self, text: str):
        self.text: str = text.lower().strip()

    def by_words(self) -> np.array:
        clean_text = re.sub(r'[.,!?;:"]', "", self.text)
        return np.array(clean_text.split())

    def by_chars(self):
        chars = np.array([], dtype=int)
        words = self.text.split()
        for index, word in enumerate(words):
            chars = np.append(chars, list(word))
            if index < len(words) - 1:
                chars = np.append(chars, " ")
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
    def __init__(self, input_dim: int, hidden_dim: int, factor: float):
        self.W_in = np.random.randn(hidden_dim, input_dim) * factor
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * factor
        self.b = np.zeros((hidden_dim,))

    def step(self, input_token, previous_hidden_state):
        input_contribution = np.dot(self.W_in, input_token)
        memory_contribution = np.dot(self.W_hidden, previous_hidden_state)
        new_hidden_state = np.tanh(input_contribution + memory_contribution + self.b)

        return new_hidden_state


class SRNN:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        factor: float = 0.05,
    ):
        self.W_in = np.random.randn(hidden_dim, input_dim) * factor
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * factor
        self.W_out = np.random.randn(output_dim, hidden_dim) * factor
        self.hidden_state = np.zeros((hidden_dim,))
        self.neuron = Neuron(input_dim, hidden_dim, factor)

    def forward(self, input_tokens):
        for token_id in input_tokens:
            input_vector = np.zeros((self.W_in.shape[1],))
            input_vector[token_id] = 1
            self.hidden_state = self.neuron.step(input_vector, self.hidden_state)

    def predict_next(self):
        raw_output = self.W_out @ self.hidden_state
        probabilities = softmax(raw_output)
        next_token_index = np.argmax(probabilities)

        return next_token_index


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)


def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def load_base_text(file: str) -> str:
    text = ""
    with open(file, "r") as f:
        text = f.read()

    return text


if __name__ == "__main__":
    text = load_base_text("./sw.txt")
    tokens = Tokenizer(text).by_words()
    vocabulary = Vocab(tokens).create()
    input_dim = len(vocabulary)
    hidden_dim = 5

    while True:
        sample = input("> ")
        if sample == "q":
            break

        test_tokens = Tokenizer(sample).by_words()
        embedded_test_tokens = Embeddings(test_tokens, vocabulary).create()

        rnn = SRNN(input_dim, hidden_dim, input_dim)
        rnn.forward(embedded_test_tokens)
        next_token = rnn.predict_next()
        next_word = get_key_by_value(vocabulary, next_token)
        print(f"{sample} {next_word}")
