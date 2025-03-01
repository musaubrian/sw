import numpy as np
import re
import sys
import os
import pickle


class Tokenizer:
    def __init__(self, text: str):
        self.text: str = text.lower().strip()

    def by_words(self) -> np.array:
        clean_text = re.sub(r'[.,!?;:"]', "", self.text)
        return np.array(clean_text.split())

    def by_chars(self):
        chars = np.array([], dtype=int)
        clean_text = re.sub(r'[.,!?;:"]', "", self.text)
        words = clean_text.split()
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
        unique_tokens = set(self.tokens)
        self.vocabulary.update(
            {token: idx for idx, token in enumerate(unique_tokens, start=1)}
        )
        return self.vocabulary


class Embeddings:
    def __init__(self, tokens: np.array, vocab: dict[str, int]):
        self.vocab = vocab
        self.tokens = tokens
        self.embeddings = np.array([], dtype=int)

    def create(self) -> np.array:
        return np.vectorize(lambda token: self.vocab.get(token, self.vocab["<UNK>"]))(
            self.tokens
        )


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
        self.hidden_states = []
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def save_model(self, file="sww.pkl"):
        with open(file, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {file}")

    @staticmethod
    def load_model(file="sww.pkl"):
        with open(file, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {file}")
        return model

    def forward(self, input_tokens, target_tokens):
        loss = 0
        self.hidden_state = np.zeros(
            (self.hidden_dim,)
        )  # Reset hidden state at beginning
        self.hidden_states = []  # Clear hidden states history

        for index, token_id in enumerate(input_tokens):
            # One-hot encode the current token
            input_vector = np.zeros((self.input_dim,))
            input_vector[target_tokens[token_id]] = 1

            self.hidden_states.append(self.hidden_state)
            self.hidden_state = self.neuron.step(input_vector, self.hidden_state)

            # Predict next token
            raw_output = np.dot(self.W_out, self.hidden_state)
            predicted_probabilities = softmax(raw_output)

            # Get the actual next token's one-hot encoded vector
            true_next_token_vector = np.zeros_like(predicted_probabilities)

            # If this is the last token, use the first token as target (for simplicity)
            next_token_idx = target_tokens[
                input_tokens[(index + 1) % len(input_tokens)]
            ]
            true_next_token_vector[next_token_idx] = 1

            loss += cross_entropy_loss(predicted_probabilities, true_next_token_vector)

        return loss / len(input_tokens)

    def backward(self, input_tokens, target_tokens, learning_rate=0.01):
        dW_in = np.zeros_like(self.W_in)
        dW_hidden = np.zeros_like(self.W_hidden)
        dW_out = np.zeros_like(self.W_out)
        delta_h_next = np.zeros_like(self.hidden_state)
        clip_threshold = 1.0

        for t in reversed(range(len(input_tokens))):
            input_vector = np.zeros((self.W_in.shape[1],))
            token_id = target_tokens[input_tokens[t]]
            input_vector[token_id] = 1

            raw_output = np.dot(self.W_out, self.hidden_states[t])
            predicted_probabilities = softmax(raw_output)

            true_next_token_vector = np.zeros_like(predicted_probabilities)
            true_next_token_vector[target_tokens[input_tokens[t]]] = 1
            delta_y = predicted_probabilities - true_next_token_vector

            # update output weights
            dW_out += np.outer(delta_y, self.hidden_states[t])

            # backprop to hidden layer
            delta_h = np.dot(self.W_out.T, delta_y) + delta_h_next
            # backprop through tanh
            tanh_deriv = 1 - self.hidden_states[t] ** 2

            norm = np.linalg.norm(delta_h)  # Compute L2 norm
            # tries to fix vanishing gradients or exploding gradients
            # by maintaining direction
            if norm > clip_threshold:
                delta_h *= clip_threshold / norm

            dW_in += np.outer(delta_h, input_vector)
            prev_hidden = np.zeros_like(self.hidden_state)
            if t > 0:
                prev_hidden = self.hidden_states[t - 1]

            outer_result = np.outer(delta_h, prev_hidden)
            dW_hidden += outer_result

            # next timestep
            delta_h_next = np.dot(self.W_hidden.T, delta_h)

        # clip gradients
        for gradient in [dW_in, dW_hidden, dW_out]:
            np.clip(gradient, -clip_threshold, clip_threshold, out=gradient)

        self.W_in -= learning_rate * dW_in
        self.W_hidden -= learning_rate * dW_hidden
        self.W_out -= learning_rate * dW_out

    def predict_next(self, input_tokens):
        self.hidden_state = np.zeros((self.W_in.shape[0],))
        # process input tokens first
        for token in input_tokens:
            input_vector = np.zeros((self.W_in.shape[1],))
            input_vector[token] = 1
            self.hidden_state = self.neuron.step(input_vector, self.hidden_state)

        raw_output = np.dot(self.W_out, self.hidden_state)
        probabilities = softmax(raw_output)

        # return top 3 predictions for variety
        top_indices = np.argsort(probabilities)[-3:][::-1]
        # randomly select from top 3 with probability proportional to softmax
        top_probs = probabilities[top_indices]
        top_probs = top_probs / np.sum(top_probs)  # Renormalize
        chosen_idx = np.random.choice(top_indices, p=top_probs)

        return chosen_idx

    def train(self, input_tokens, target_tokens, epochs=100, learning_rate=0.01):
        prev_losses = [float("inf")] * 3  # Track last three losses

        for epoch in range(epochs):
            loss = self.forward(input_tokens, target_tokens)
            self.backward(input_tokens, target_tokens, learning_rate)

            if epoch % 50 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

            if loss > max(prev_losses):
                print("Loss increasing - stopping early")
                break

            prev_losses = prev_losses[1:] + [loss]  # Shift losses

            # learning rate decay every 10 epochs
            if epoch % 10 == 0:
                learning_rate = max(learning_rate * 0.9, 0.001)
                print(f"Reducing learning rate to {learning_rate:.6f}")


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)


def cross_entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0) issues
    return -np.sum(y_true * np.log(y_pred))


def load_base_text(file: str) -> str:
    text = ""
    with open(file, "r") as f:
        text = f.read()

    return text


if __name__ == "__main__":
    sww_pkl_file = "sww.pkl"

    text = load_base_text("sw.txt")
    tokens = Tokenizer(text).by_words()
    vocabulary = Vocab(tokens).create()
    input_dim = len(vocabulary)
    hidden_dim = 150
    srnn = SRNN(input_dim, hidden_dim, input_dim)

    args = sys.argv
    if len(args) != 2:
        print("Expected at least one arg [train|inf]")
        sys.exit(1)

    if args[1] == "train":
        if os.path.isfile(sww_pkl_file):
            srnn = SRNN.load_model(sww_pkl_file)

        srnn.train(tokens, vocabulary, 500)
        srnn.save_model(sww_pkl_file)
    elif args[1] == "inf":
        srnn.load_model(sww_pkl_file)
        reverse_vocab = {v: k for k, v in vocabulary.items()}

        while True:
            sample = input("> ")
            if sample == "q":
                break

            for count in range(5):
                test_tokens = Tokenizer(sample).by_words()
                embedded_test_tokens = Embeddings(test_tokens, vocabulary).create()

                next_token = srnn.predict_next(embedded_test_tokens)
                next_word = reverse_vocab.get(next_token, "<UNK>")
                sample = f"{sample} {next_word}"

            print(f"{sample}")
