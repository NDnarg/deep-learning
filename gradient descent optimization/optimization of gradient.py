import numpy as np
import matplotlib.pyplot as plt
import opt_utils

from C2week2.gd import initialize_velocity, initialize_adam, random_mini_batches, update_parameters_with_gd, \
    update_parameters_with_momentun, update_parameters_with_adam

train_X, train_Y = opt_utils.load_dataset(is_plot=False)


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):

    L = len(layers_dims)
    costs = []
    t = 0  # 每学习完一个minibatch就增加1
    seed = 10  # 随机种子

    # 初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)

    # 选择优化器
    if optimizer == "gd":
        pass  # 不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 使用动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 使用Adam优化
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)

    # 开始学习
    for i in range(num_epochs):
        # 定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # 前向传播
            A3, cache = opt_utils.forward_propagation(minibatch_X, parameters)

            # 计算误差
            cost = opt_utils.compute_cost(A3, minibatch_Y)

            # 反向传播
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)

            # 更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,epsilon)
        # 记录误差值
        if i % 100 == 0:
            costs.append(cost)
            # 是否打印误差值
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
    # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters
"""
#使用普通的梯度下降
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, optimizer="gd",is_plot=True)
#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)
#绘制分类图
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
"""

"""
#使用动量的梯度下降
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, beta=0.9,optimizer="momentum",is_plot=True)
#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)
#绘制分类图
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
"""


#使用Adam优化的梯度下降
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam",is_plot=True)
#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)
#绘制分类图
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

