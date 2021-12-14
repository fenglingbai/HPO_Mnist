import math
import numpy as np
from matplotlib import pyplot as plt
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor


from CNN_OPT import single_opt
# uppper confidence bound
# beta = 1
def acquisition(X, surrogate):
    """
    采样函数
    :param X:
    :param surrogate:
    :return:
    """
    yhat, std = surrogate.predict(X, return_std=True)
    yhat = yhat.flatten()
    upper = yhat + std
    max_at = np.argmax(upper)
    return X[max_at]

def dict_flatten(spacedict):
    """
    将待定的参数阈值转化为所有的可选项
    :param spacedict:
    :return:
    """
    para_all = []
    for key in spacedict.keys():
        para_all_len = len(para_all)
        if para_all_len == 0:
            for para_item in spacedict[key]:
                para_all.append({key:para_item})
        else:
            for add_item in range(para_all_len):
                temp_dict = para_all[0]
                para_all = para_all[1:]
                for para_item in spacedict[key]:
                    temp_dict_add = temp_dict.copy()
                    temp_dict_add.update({key: para_item})
                    para_all.append(temp_dict_add)
    return para_all


def plot(X, xsamples, ysamples, yhat, std, new_x, new_y, i, savesetting=False):
    """
    绘图函数
    :param X:
    :param xsamples:
    :param ysamples:
    :param yhat:
    :param std:
    :param new_x:
    :param new_y:
    :param i:
    :return:
    """

    plt.scatter(xsamples, ysamples, label='explored samples')
    plt.plot(X, yhat, label='gussian process - mean', c='g')
    plt.plot(X, yhat + std, label='gussian process - upper/lower bound', c='g', linestyle='--', )
    plt.plot(X, yhat - std, c='g', linestyle='--', )
    plt.scatter([new_x], [new_y], label='next sample', c='r')
    plt.legend()
    plt.title(f'Iteration {i}')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')  # 设定子图212的Y轴lable为amplitude
    if savesetting:
        plt.savefig(f'Iteration {i}' + r'.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
    plt.show()
    plt.close('all')  # 关闭绘图对象，释放绘图资源
# spacedict = {'gjy':[1,2,3],
#              'zxy':[3,2,1],
#              'lll':['23','12']}
# ans = dict_flatten(spacedict)
# print('ok')

# def para_index_compute(spacedict):
#     keyset = []
#     keyset_len = []
#     for key in spacedict.keys():
#         keyset.append(key)
#         keyset_len.append(len(spacedict[key]))
#     para_index = np.arange(keyset_len[0])
#     para_index = para_index[:, np.newaxis]
#     for dim in range(1, len(para_index)):
#         pass
#     return False
#
# ans2 = para_index_compute(spacedict)
# print('ok')

def func(learning_rate):
    loss_test, acc_test = single_opt(learning_rate[0], EPOCHS=1)
    return np.array([acc_test])


# def bayesian_GPR(func, acquisition, max_evals=4):
if __name__ == "__main__":
    # 代理函数，使用高斯过程
    surrogate = GaussianProcessRegressor()
    # sample of the domain [0,1]
    X = np.arange(0, 1, 0.01)
    X = X.reshape(-1, 1)
    xsamples = np.array([[0], [1]])
    ysamples = np.array([func(x) for x in xsamples])
    for i in range(8):
        # X=X.reshape(-1,1)
        # xsamples=xsamples.reshape(-1, 1)
        # ysamples=ysamples.reshape(-1, 1)
        # step 1 update the surrogate function
        surrogate.fit(xsamples, ysamples)
        yhat, std = surrogate.predict(X, return_std=True)
        std = std.reshape(-1, 1)
        yhat = yhat.reshape(-1, 1)
        # step
        new_x = acquisition(X, surrogate)
        new_y = func(new_x)
        plot(X, xsamples, ysamples, yhat, std, new_x, new_y, i)
        # print(f'max y is {max(ysamples.flatten())}')
        xsamples = np.vstack((xsamples, new_x))
        ysamples = np.vstack((ysamples, new_y))

    plt.plot(xsamples, label='xsamples curve', c='r')
    plt.legend()
    plt.title(r'Learning rate curve')
    plt.xlabel('Number of iterations')
    plt.ylabel('Learning rate')  # 设定子图212的Y轴lable为amplitude
    plt.savefig(r'Learning rate curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
    plt.show()
    plt.close('all')  # 关闭绘图对象，释放绘图资源

    plt.plot(ysamples, label='ysamples curve', c='b')
    plt.legend()
    plt.title(r'Accuracy curve')
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')  # 设定子图212的Y轴lable为amplitude
    plt.savefig(r'Accuracy curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
    plt.show()
    plt.close('all')  # 关闭绘图对象，释放绘图资源
    print('ok')
