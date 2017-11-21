import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import glob
import argparse
import json
# from tqdm import tqdm

from model import model
from keras.callbacks import EarlyStopping


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
    parser.add_argument('--train_dir',
                        type=str,
                        default="data/data/train/",
                        help='Training file')
    parser.add_argument('--dev_dir',
                        type=str,
                        default="data/data/dev/",
                        help='dev file')
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
    x = [[], [], [], [], []]
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
                    a_q = np.concatenate((av, qv), axis=0)
                    a_q.resize((1024, 300), refcheck=False)
                except:
                    print(q, qv)
                    continue
                for i, o in enumerate(os):
                    ops = get_w2v(o)
                    ops.resize((256, 300), refcheck=False)
                    x[i+1].append(ops)
                x[0].append(a_q)
                y.append(A2onehot(ans))
    nx = [np.array(ob) for ob in x]
    return nx, np.array(y)


def load_bach(dir, batch_size=32):
    def reset():
        x = []
        y = []
        return x, y, 0
    question_list = glob.glob(dir+'*/*.txt', recursive=True)
    x, y, count = reset()
    for q in question_list:
        with open(q, 'r')as f:
            dic = json.load(f)
            av = get_w2v(dic["article"])
            av.resize((768, 300), refcheck=False)
            for que, os, ans in zip(dic["questions"], dic["options"], dic["answers"]):
                count += 1
                if count > batch_size:
                    yield (np.array(x), np.array(y))
                    x, y, count = reset()
                qv = get_w2v(que)
                qv.resize((256, 300), refcheck=False)
                qtext = np.concatenate((av, qv), axis=0)
                for i, o in enumerate(os):
                    ops = get_w2v(o)
                    ops.resize((256, 300), refcheck=False)
                    qtext = np.concatenate((qtext, ops), axis=0)
                x.append(qtext)
                y.append(A2onehot(ans))


if __name__ == '__main__':
    args = get_args()
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
            'data/GoogleNews-vectors-negative300.bin', binary=True)

    # model.fit(x_dic, y_train, batch_size=32, epochs=10)  # , callbacks=[early_stopping])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit_generator(load_bach(args.train_dir),
                        steps_per_epoch=3000, epochs=100,
                        validation_data=load_bach(args.dev_dir, 1),
                        validation_steps=120,
                        use_multiprocessing=True,
                        callbacks=[early_stopping])
    model.save('data/model.keras')

    #score = model2.evaluate(x_test, y_test, batch_size=16)
