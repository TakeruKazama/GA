import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import glob
import argparse
import json
from tqdm import tqdm
import pickle

from model import model


def A2onehot(c):

    if c == 'A':
        r = [1, 0, 0, 0]
    elif c == 'B':
        r = [0, 1, 0, 0]
    elif c == 'C':
        r = [0, 0, 1, 0]
    elif c == 'D':
        r = [0, 0, 0, 1]
    else:
        raise KeyError("not expected key!")
    return np.array(r)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.register('type', 'bool', str2bool)
    parser.add_argument('--train_file',
                        type=str,
                        default="data/data/train/",
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
    x = [[],[],[],[],[]]
    y = []
    question_list = glob.glob(dir+'*/*.txt', recursive=True)
    for q in tqdm(question_list):
        with open(q, 'r')as f:
            dic = json.load(f)
            av = get_w2v(dic["article"])
            for que, os, ans in zip(dic["questions"], dic["options"], dic["answers"]):
                qv = get_w2v(que)
                # print(av.shape, qv.shape)
                try:
                    a_q = np.array([list(av) + list(qv)])
                except:
                    print(q, qv)
                    continue
                for i, o in enumerate(os):
                    a_q.append(np.array(get_w2v(o)))
                x.append(np.array(a_q))
                y.append(A2onehot(ans))
    #nx = [np.array(ob) for ob in x]
    return np.array(x), np.array(y)

if __name__ == '__main__':
    args = get_args()
    try:
        with open('xtra.pickle', 'rb') as f:
            x_train = pickle.load(f)
        with open('ytra.pickle', 'rb') as f:
            y_train = pickle.load(f)
    except:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(
            'data/GoogleNews-vectors-negative300.bin', binary=True)
        x_train, y_train = load_data(args.train_file)
        with open('xtra.pickle', 'wb') as f:
            pickle.dump(x_train, f)
        with open('ytra.pickle', 'wb') as f:
            pickle.dump(y_train, f)

    print("vec done", x_train.shape)
    x_dic = {"argm": x_train[0], "o1": x_train[1], "o2": x_train[2], "o3": x_train[3], "o4": x_train[4]}
    model.fit(x_dic,
              y_train,
              batch_size=32, epochs=10)  # , callbacks=[early_stopping])
    with open('data/model.keras', mode='wb') as f:
        model.save(f)

    #score = model2.evaluate(x_test, y_test, batch_size=16)
