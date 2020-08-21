import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

from C4week1 import cnn_utils
from C4week1.tf_utils import create_placeholders, initialize_parameters, forward_propagation, compute_cost

np.random.seed(1)
X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = cnn_utils.load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T

conv_layers = {}


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True, isPlot=True):
    """
    使用TensorFlow实现三层的卷积神经网络
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    参数：
        X_train - 训练数据，维度为(None, 64, 64, 3)
        Y_train - 训练数据对应的标签，维度为(None, n_y = 6)
        X_test - 测试数据，维度为(None, 64, 64, 3)
        Y_test - 训练数据对应的标签，维度为(None, n_y = 6)
        learning_rate - 学习率
        num_epochs - 遍历整个数据集的次数
        minibatch_size - 每个小批量数据块的大小
        print_cost - 是否打印成本值，每遍历100次整个数据集打印一次
        isPlot - 是否绘制图谱

    返回：
        train_accuracy - 实数，训练集的准确度
        test_accuracy - 实数，测试集的准确度
        parameters - 学习后的参数
    """
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)  # 确保你的数据和我一样
    seed = 3  # 指定numpy的随机种子
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    # 为当前维度创建占位符
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播，由于框架已经实现了反向传播，我们只需要选择一个优化器就行了
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 全局初始化所有变量
    init = tf.global_variables_initializer()

    # 开始运行
    with tf.Session() as sess:
        # 初始化参数
        sess.run(init)
        # 开始遍历数据集
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)  # 获取数据块的数量
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # 对每个数据块进行处理
            for minibatch in minibatches:
                # 选择一个数据块
                (minibatch_X, minibatch_Y) = minibatch
                # 最小化这个数据块的成本
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches

            # 是否打印成本
            if print_cost:
                # 每5代打印一次
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

            # 记录成本
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

        # 数据处理完毕，绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 开始预测数据
        ## 计算当前的预测情况
        predict_op = tf.arg_max(Z3, 1)
        corrent_prediction = tf.equal(predict_op, tf.arg_max(Y, 1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))

        return (train_accuracy, test_accuary, parameters)
_, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epochs=150)
