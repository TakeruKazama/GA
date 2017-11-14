import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import glob
import argparse
import json

import model

w2v = gensim.models.KeyedVectors.load_word2vec_format(
    'data/GoogleNews-vectors-negative300.bin', binary=True)


def A2onehot(c):
    if c == 'A':
        return [1, 0, 0, 0]
    elif c == 'B':
        return [0, 1, 0, 0]
    elif c == 'C':
        return [0, 0, 1, 0]
    elif c == 'D':
        return [0, 0, 0, 1]
    else:
        raise KeyError("not expected key!")


def get_args():
    parser = argparse.ArgumentParser()
    # parser.register('type', 'bool', str2bool)
    parser.add_argument('-train_file',
                        type=str,
                        default="data/data/train",
                        help='Training file')
    return parser.parse_args()


def get_w2v(text):
    text = word_tokenize(text)
    word2vec_array = []
    for word in text:
        try:
            word2vec_array.append(w2v[word])
        except KeyError:
            pass
    return np.array(word2vec_array)


def load_data(dir):
    x = []
    y = []
    question_list = glob.glob(dir+'/*/*.txt', recursive=True)
    for i, q in enumerate(question_list):
        with open(q, 'r')as f:
            dic = json.load(f)
            av = get_w2v(dic["article"])
            for q, os, ans in zip(dic["questions"], dic["options"], dic["answers"]):
                qv = get_w2v(dic[q])
                t_list = [[av, qv]]
                for o in os:
                    t_list.append(get_w2v(o))
                x.append(t_list)
                y.append(A2onehot(ans))
    return x, y


if __name__ == '__main__':
    args = get_args()
    x_train, y_train = load_data(args.train_file)

    model.model2.fit(x_train, y_train, batch_size=32, epochs=10)  # , callbacks=[early_stopping])
    #score = model2.evaluate(x_test, y_test, batch_size=16)
