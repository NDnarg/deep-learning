import numpy as np
import emoji
import matplotlib.pyplot as plt
from C5week2 import emo_utils

X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')
maxLen = len(max(X_train, key=len).split())
Y_oh_train = emo_utils.convert_to_one_hot(Y_train, C=5)
Y_oh_test = emo_utils.convert_to_one_hot(Y_test, C=5)
word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')


def sentence_to_avg(sentence, word_to_vec_map):

    # 第一步：分割句子，转换为列表。
    words = sentence.lower().split()

    # 初始化均值词向量
    avg = np.zeros(50, )

    # 第二步：对词向量取平均。
    for w in words:
        avg += word_to_vec_map[w]
    avg = np.divide(avg, len(words))

    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):

    np.random.seed(1)

    # 定义训练数量
    m = Y.shape[0]
    n_y = 5
    n_h = 50

    # 使用Xavier初始化参数
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # 将Y转换成独热编码
    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)

    # 优化循环
    for t in range(num_iterations):
        for i in range(m):
            # 获取第i个训练样本的均值
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # 前向传播
            z = np.dot(W, avg) + b
            a = emo_utils.softmax(z)

            # 计算第i个训练的损失
            cost = -np.sum(Y_oh[i] * np.log(a))

            # 计算梯度
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # 更新参数
            W = W - learning_rate * dW
            b = b - learning_rate * db
        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t, cost=cost))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b
pred, W, b = model(X_train, Y_train, word_to_vec_map)
print("=====训练集====")
pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
print("=====测试集====")
pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = emo_utils.predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
emo_utils.print_predictions(X_my_sentences, pred)
