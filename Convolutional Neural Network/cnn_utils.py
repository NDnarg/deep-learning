import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def zero_pad(X, pad):


    X_paded = np.pad(X, (
        (0, 0),  # 样本数，不填充
        (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0, 0)),  # 通道数，不填充
                     'constant', constant_values=0)  # 连续一样的值填充

    return X_paded


def conv_single_step(a_slice_prev, W, b):


    s = np.multiply(a_slice_prev, W) + b

    Z = np.sum(s)

    return Z


def conv_forward(A_prev, W, b, hparameters):

    # 获取来自上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行板除
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # 使用0来初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))

    # 通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    # 自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 执行单步卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    # 数据处理完毕，验证数据格式是否正确
    assert (Z.shape == (m, n_H, n_W, n_C))

    # 存储一些缓存值，以便于反向传播使用
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)


def pool_forward(A_prev, hparameters, mode="max"):

    # 获取输入数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取超参数的信息
    f = hparameters["f"]
    stride = hparameters["stride"]

    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    # 初始化输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # 遍历样本
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 定位完毕，开始切割
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # 对切片进行池化操作
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    # 池化完毕，校验数据格式
    assert (A.shape == (m, n_H, n_W, n_C))

    # 校验完毕，开始存储用于反向传播的值
    cache = (A_prev, hparameters)

    return A, cache


def conv_backward(dZ, cache):

    # 获取cache的值
    (A_prev, W, b, hparameters) = cache

    # 获取A_prev的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取dZ的基本信息
    (m, n_H, n_W, n_C) = dZ.shape

    # 获取权值的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取hparaeters的值
    pad = hparameters["pad"]
    stride = hparameters["stride"]

    # 初始化各个梯度的结构
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    # 现在处理数据
    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # 数据处理完毕，验证数据格式是否正确
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return (dA_prev, dW, db)


def create_mask_from_window(x):
    """
    从输入矩阵中创建掩码，以保存最大值的矩阵的位置。

    参数：
        x - 一个维度为(f,f)的矩阵

    返回：
        mask - 包含x的最大值的位置的矩阵
    """
    mask = x == np.max(x)

    return mask


def distribute_value(dz, shape):
    """
    给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。

    参数：
        dz - 输入的实数
        shape - 元组，两个值，分别为n_H , n_W

    返回：
        a - 已经分配好了值的矩阵，里面的值全部一样。

    """
    # 获取矩阵的大小
    (n_H, n_W) = shape

    # 计算平均值
    average = dz / (n_H * n_W)

    # 填充入矩阵
    a = np.ones(shape) * average

    return a


def pool_backward(dA, cache, mode="max"):
    """
    实现池化层的反向传播

    参数:
        dA - 池化层的输出的梯度，和池化层的输出的维度一样
        cache - 池化层前向传播时所存储的参数。
        mode - 模式选择，【"max" | "average"】

    返回：
        dA_prev - 池化层的输入的梯度，和A_prev的维度相同

    """
    # 获取cache中的值
    (A_prev, hparaeters) = cache

    # 获取hparaeters的值
    f = hparaeters["f"]
    stride = hparaeters["stride"]

    # 获取A_prev和dA的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    # 初始化输出的结构
    dA_prev = np.zeros_like(A_prev)

    # 开始处理数据
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 选择反向传播的计算方式
                    if mode == "max":
                        # 开始切片
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 创建掩码
                        mask = create_mask_from_window(a_prev_slice)
                        # 计算dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":
                        # 获取dA的值
                        da = dA[i, h, w, c]
                        # 定义过滤器大小
                        shape = (f, f)
                        # 平均分配
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    # 数据处理完毕，开始验证格式
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    return Z3
def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    x = tf.placeholder("float", [12288, 1])
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
    return prediction