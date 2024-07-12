# coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SentimentRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = hidden.squeeze(0)
        y_pred = self.fc(x)
        if y is not None:
            return self.loss_fn(y_pred, y)
        else:
            return y_pred

def build_vocab(sentences):
    vocab = {"pad": 0, "unk": 1}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode_sentence(sentence, vocab):
    return [vocab.get(word, vocab["unk"]) for word in sentence.split()]

def build_dataset(sentences, labels, vocab):
    encoded_sentences = [encode_sentence(sentence, vocab) for sentence in sentences]
    max_length = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [vocab["pad"]] * (max_length - len(sentence)) for sentence in encoded_sentences]
    return torch.LongTensor(padded_sentences), torch.LongTensor(labels)

def evaluate(model, dataset, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(dataset[0]), batch_size):
            x_batch = dataset[0][i:i+batch_size]
            y_batch = dataset[1][i:i+batch_size]
            y_pred = model(x_batch)
            correct += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

def main():
    sentences = [
        "I love this movie",
        "I hate this movie",
        "This film is great",
        "This film is terrible",
        "I enjoyed this film",
        "I disliked this film",
        "The movie was fantastic",
        "The movie was awful",
        "I like this movie",
        "I don't like this movie"
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    vocab = build_vocab(sentences)
    dataset = build_dataset(sentences, labels, vocab)

    embedding_dim = 50
    hidden_dim = 50
    output_dim = 2
    model = SentimentRNNModel(len(vocab), embedding_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    batch_size = 2

    for epoch in range(epochs):
        model.train()
        losses = []
        for i in range(0, len(dataset[0]), batch_size):
            x_batch = dataset[0][i:i+batch_size]
            y_batch = dataset[1][i:i+batch_size]
            optimizer.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        accuracy = evaluate(model, dataset)
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses)}, Accuracy: {accuracy}")

    torch.save(model.state_dict(), "sentiment_model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))

def predict(model_path, vocab_path, sentences):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    embedding_dim = 50
    hidden_dim = 50
    output_dim = 2
    model = SentimentRNNModel(len(vocab), embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    encoded_sentences = [encode_sentence(sentence, vocab) for sentence in sentences]
    max_length = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [vocab["pad"]] * (max_length - len(sentence)) for sentence in encoded_sentences]
    inputs = torch.LongTensor(padded_sentences)

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        for sentence, prediction in zip(sentences, predictions):
            sentiment = "positive" if prediction.item() == 1 else "negative"
            print(f"Sentence: '{sentence}' is {sentiment}")

if __name__ == "__main__":
    main()
    test_sentences = [
        "I love this film",
        "I hate this film",
        "This movie is fantastic",
        "This movie is terrible"
    ]
    predict("sentiment_model.pth", "vocab.json", test_sentences)
