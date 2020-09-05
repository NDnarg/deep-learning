import numpy as np
from C5week2 import w2v_utils


def cosine_similarity(u, v):

    distance = 0

    # 计算u与v的内积
    dot = np.dot(u, v)

    # 计算u的L2范数
    norm_u = np.sqrt(np.sum(np.power(u, 2)))

    # 计算v的L2范数
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    # 根据公式1计算余弦相似度
    cosine_similarity = np.divide(dot, norm_u * norm_v)

    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):

    # 把单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # 获取对应单词的词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    # 获取全部的单词
    words = word_to_vec_map.keys()

    # 将max_cosine_sim初始化为一个比较大的负数
    max_cosine_sim = -100
    best_word = None

    # 遍历整个数据集
    for word in words:
        # 要避免匹配到输入的数据
        if word in [word_a, word_b, word_c]:
            continue
        # 计算余弦相似度
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word


words, word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt')
#test
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))

triads_to_try = [('small', 'smaller', 'big')]
for triad in triads_to_try:
    print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
