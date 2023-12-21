from torch import nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from gensim import downloader
import numpy as np
import re
from torch.optim import Adam
import pickle
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    sentences_tags = []
    sentence_tags = []
    words = []
    word_to_label = {}

    for line in lines:
        line = re.sub(r'\ufeff', '', line)
        if line == '\t\n' or line == '\n':
            sentences.append(sentence)
            sentences_tags.append(sentence_tags)
            sentence = []
            sentence_tags = []
        else:
            word, tag = line.split('\t')
            tag = 0 if tag[:-1] == "O" else 1
            sentence.append(word)
            sentence_tags.append(tag)
            words.append(word)
            word_to_label[word] = tag

    return sentences, sentences_tags, words, word_to_label


def build_set(sentences, model1, model2, sentences_tags):
    set_data = []
    set_tags = []
    all_tags = []
    representation_dict = {}

    for sentence, tags in zip(sentences, sentences_tags):
            all_tags += tags
            tokenized_sentence = []
            for word in sentence:
                if word not in representation_dict:
                    if word not in model1.key_to_index:
                        word_vec_1 = torch.as_tensor(model1['oov'].tolist())
                    else:
                        word_vec_1 = torch.as_tensor(model1[word].tolist())

                    if word not in model2.key_to_index:
                        word_vec_2 = torch.zeros(model2.vector_size)
                    else:
                        word_vec_2 = torch.as_tensor(model2[word].tolist())

                    final_vec = torch.cat((word_vec_1, word_vec_2))
                    representation_dict[word] = torch.cat((word_vec_1, word_vec_2))
                    tokenized_sentence.append(final_vec)
                else:
                    tokenized_sentence.append(representation_dict[word])

            final_tokenized_sen = torch.stack(tokenized_sentence)
            set_data.append(final_tokenized_sen)
            set_tags.append(torch.LongTensor(tags))

    return set_data, set_tags, all_tags


class NerNN(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=128):
        super(NerNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
        self.third_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Tanh()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.activation(x)
        x = self.third_layer(x)

        if labels is None:
            return x, None

        loss = self.loss(x, labels)
        return x, loss


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, model1, model1_name, model2, model2_name):
        self.file_path = file_path
        self.sentences, original_tags, words, word_to_label = load_data(file_path)
        self.vector_dim = int(re.findall(r'\d+', model1_name)[-1]) + int(re.findall(r'\d+', model2_name)[-1])
        self.tokenized_sen, self.tags, self.all_tags = build_set(self.sentences, model1, model2, original_tags)


def plot(train_f1, dev_f1,epochs):
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(epochs, train_f1, color = "blue", label = "train f1")
    ax.plot(epochs, dev_f1, color = "green", label = "dev f1")
    ax.plot(epochs, [0.5 for i in range(len(epochs))], color="red", label = "threshold")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    plt.xlabel("epochs")
    plt.ylabel("f1 score")
    plt.title("Train and Test F1 scores")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def learn_and_predict(model, data_sets, optimizer, num_epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_F1, dev_F1 = [], []
    model.to(device)
    max_f1 = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        print("-" * 30)
        for phase in ["train", "dev"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            labels, preds = [], []
            dataset = data_sets[phase]
            for sentence, sentence_tags in zip(dataset.tokenized_sen, dataset.tags):
                if phase == "train":
                    outputs, loss = model(sentence, sentence_tags)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        outputs, loss = model(sentence, sentence_tags)

                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += sentence_tags.cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()

            epoch_f1 = f1_score(labels, preds)
            if phase == "train":
                train_F1.append(epoch_f1)
            else:
                dev_F1.append(epoch_f1)
            print(f"{phase} F1: {epoch_f1}")

            # update max f1 score
            if phase == "dev" and epoch_f1 > max_f1:
                max_f1 = epoch_f1
        print()

    print(f"Max F1: {max_f1:.4f}")

    plot(train_F1, dev_F1, [i + 1 for i in range(num_epochs)])


def main(num_epochs: int):
    model1 = downloader.load('glove-twitter-50')
    model2 = downloader.load('word2vec-google-news-300')
    model1_name = "glove-twitter-50"
    model2_name = "word2vec-google-news-300"
    train_set = NerDataset("data/train.tagged", model1, model1_name, model2, model2_name)
    test_set = NerDataset("data/dev.tagged", model1, model1_name, model2, model2_name)
    nn_model = NerNN(vec_dim=train_set.vector_dim, num_classes=2)
    optimizer = Adam(params=nn_model.parameters(), lr=0.01)
    datasets = {"train": train_set, "dev": test_set}
    learn_and_predict(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=num_epochs)


if __name__ == "__main__":
    main(5)

