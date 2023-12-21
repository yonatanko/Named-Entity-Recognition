from gensim import downloader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import re
import matplotlib.pyplot as plt
import pandas as pd


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    all_tags = []
    sentence_tags = []
    words = []
    word_to_label = {}

    for line in lines:
        line = re.sub(r'\ufeff', '', line)
        if line == '\t\n' or line == '\n':
            sentences.append(sentence)
            all_tags.append(sentence_tags)
            sentence = []
            sentence_tags = []
        else:
            word, tag = line.split('\t')
            tag = 0 if tag[:-1] == "O" else 1
            sentence.append(word)
            sentence_tags.append(tag)
            words.append(word)
            word_to_label[word] = tag

    return sentences, all_tags, words, word_to_label


def build_set(words, model, word_to_label, length):
    set_data = []
    set_tags = []

    for word in words:
        if word not in model.key_to_index:
            word_vec = np.zeros(length)
        else:
            word_vec = model[word]

        set_data.append(word_vec)
        set_tags.append(word_to_label[word])

    return set_data, set_tags


def train_and_predict(model, k, length):
    # Load training data
    sentences, all_tags, words, word_to_label = load_data('data/train.tagged')
    train_all_words = set(words)

    # generate train set
    train_set, train_tags = build_set(train_all_words, model, word_to_label, length)
    # train knn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, train_tags)

    # load test data
    test_sentences, test_tags, test_words, test_word_to_label = load_data('data/dev.tagged')
    test_all_words = set(test_words)

    # generate test set
    test_set, test_tags = build_set(test_all_words, model, test_word_to_label, length)

    # predict
    predictions = knn.predict(test_set)

    # calculate F1 score
    f1_score_value = f1_score(test_tags, predictions)
    return f1_score_value


def plot_f1(models_f1, k_list):
    fig = plt.figure()
    ax = plt.subplot(111)

    for model_name in models_f1:
        ax.scatter(k_list, models_f1[model_name], label=model_name)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.plot(k_list, [0.5]*len(k_list), color = "red",  label='threshold')
    plt.xlabel('k')
    plt.ylabel('F1 score')
    plt.title('F1 score for different k and models \n of word embeddings', fontsize=15, fontweight='bold')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def build_dataframe_of_f1(models_f1, k_list):
    df = pd.DataFrame(models_f1, index=k_list)
    # show dataframe
    print(df)


# def main():
#     models_f1 = {}
#     for k in [1, 3, 5]:
#         print(f"num of neighbors: {k}")
#
#         for model_name in ['word2vec-google-news-300','glove-wiki-gigaword-50','glove-wiki-gigaword-100','glove-wiki-gigaword-200',
#                            'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']:
#
#             if model_name not in models_f1:
#                 models_f1[model_name] = []
#
#             print(f"model: {model_name}")
#             # extract number from model name
#             dim = int(re.findall(r'\d+', model_name)[-1])
#             model = downloader.load(model_name)
#             f1_model = train_and_predict(model, k, dim)
#             models_f1[model_name].append(f1_model)
#
#
#     plot_f1(models_f1, [1, 3, 5])
#     build_dataframe_of_f1(models_f1, [1, 3, 5])


def main():
    models_f1 = {}
    models = {}
    for model_name in ['word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
                       'glove-wiki-gigaword-200',
                       'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']:
        models_f1[model_name] = []
        models[model_name] = downloader.load(model_name)
    for k in [1, 3, 5]:
        print(f"num of neighbors: {k}")

        for model_name in ['word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
                           'glove-wiki-gigaword-200',
                           'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']:
            print(f"model: {model_name}")
            # extract number from model name
            dim = int(re.findall(r'\d+', model_name)[-1])
            model = models[model_name]
            f1_model = train_and_predict(model, k, dim)
            models_f1[model_name].append(f1_model)

    plot_f1(models_f1, [1, 3, 5])
    build_dataframe_of_f1(models_f1, [1, 3, 5])


if __name__ == '__main__':
    main()
