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


def build_set(sentences, model1, model2, sentences_tags, underSample_threshold, overSample_threshold, if_train):
    set_data = []
    set_tags = []
    all_tags = []
    representation_dict = {}
    counter_removed = 0
    counter_added = 0

    for sentence, tags in zip(sentences, sentences_tags):
        if tags.count(1) <= underSample_threshold and if_train:
            counter_removed += 1
        else:
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
            if if_train:
                if tags.count(1) >= overSample_threshold:
                    for i in range(2):
                        counter_added += 1
                        set_data.append(final_tokenized_sen)
                        set_tags.append(torch.as_tensor(tags))
                        all_tags += tags

    print(f"dropped: {counter_removed} sentences")
    print(f"added: {counter_added} sentences")
    print(f"1 ratio: {all_tags.count(1) / len(all_tags)}")
    print(f"0 ratio: {all_tags.count(0) / len(all_tags)} \n")

    return set_data, set_tags, all_tags



class bi_LSTM_NER(nn.Module):
    def __init__(self, vec_dim, num_classes, weights, hidden_dim=64):
        super(bi_LSTM_NER, self).__init__()
        self.lstm = nn.LSTM(vec_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.activation = nn.Tanh()
        self.fc_2 = nn.Linear(hidden_dim * 2, num_classes)
        self.weights = weights
        self.loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')

    def forward(self, input_ids, labels=None):
        input_ids_3d = input_ids.unsqueeze(1)  # transform 2d input_ids to 3d input_ids
        x, _ = self.lstm(input_ids_3d)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        x = x.squeeze(1)  # transform 3d output to 2d output
        if labels is None:
            return x, None

        loss = self.loss(x, labels)
        return x, loss



class NerDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, model1, model1_name, model2, model2_name, underSample_threshold, overSample_threshold,
                 if_train):
        self.file_path = file_path
        self.sentences, tags, words, word_to_label = load_data(file_path)
        self.vector_dim = int(re.findall(r'\d+', model1_name)[-1]) + int(re.findall(r'\d+', model2_name)[-1])
        self.tokenized_sen, self.tags, self.all_tags = build_set(self.sentences, model1, model2, tags,
                                                                 underSample_threshold, overSample_threshold, if_train)


def plot(train_f1, dev_f1, epochs):
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(epochs, train_f1, color="blue", label="train f1")
    ax.plot(epochs, dev_f1, color="green", label="dev f1")
    ax.plot(epochs, [0.5 for i in range(len(epochs))], color="red", label="threshold")
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

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    plot(train_F1, dev_F1, [i + 1 for i in range(num_epochs)])


def main(num_epochs, underSample_threshold, overSample_threshold):
    model1 = downloader.load('glove-twitter-50')
    model2 = downloader.load('word2vec-google-news-300')
    model1_name = "glove-twitter-50"
    model2_name = "word2vec-google-news-300"

    print("Train Data:")
    train_set = NerDataset("data/train.tagged", model1, model1_name, model2, model2_name, underSample_threshold,
                           overSample_threshold, True)
    print("Dev Data:")
    test_set = NerDataset("data/dev.tagged", model1, model1_name, model2, model2_name, underSample_threshold,
                          overSample_threshold, False)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_set.all_tags),
                                         y=np.array(train_set.all_tags))
    print(f"Classes weights: {class_weights}")
    class_weights = torch.FloatTensor(class_weights)
    nn_model = bi_LSTM_NER(vec_dim=train_set.vector_dim, weights=class_weights, num_classes=2)

    optimizer = Adam(params=nn_model.parameters(), lr=0.01)
    datasets = {"train": train_set, "dev": test_set}
    learn_and_predict(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=num_epochs)


if __name__ == '__main__':
    main(5,0,8)
