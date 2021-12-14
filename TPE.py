import os
import numpy as np
from matplotlib import pyplot as plt

from CNN_OPT import objective
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK




def func(argsDict):
    outdict = {}
    outdict["BATCH_SIZE"] = int(argsDict["BATCH_SIZE"] * 5 + 50)
    outdict['learning_rate'] = argsDict['learning_rate'] * 0.1
    outdict["Conv2d_1_1"] = argsDict["Conv2d_1_1"] * 8 + 16
    outdict["Conv2d_1_2"] = argsDict["Conv2d_1_2"] * 8 + 16
    outdict["Conv2d_2_1"] = argsDict["Conv2d_2_1"] * 16 + 32
    outdict["Conv2d_2_2"] = argsDict["Conv2d_2_2"] * 16 + 32
    loss_test, acc_test = objective(argsDict=outdict)


    if not os.path.exists('log.npy'):
        np.save("log.npy", np.array([[outdict["BATCH_SIZE"],
                                      outdict['learning_rate'],
                                      outdict['Conv2d_1_1'],
                                      outdict['Conv2d_1_2'],
                                      outdict['Conv2d_2_1'],
                                      outdict['Conv2d_2_2'],
                                      loss_test,
                                      acc_test]]))
    else:
        old_log = np.load("log.npy")
        new_log = np.array([[outdict["BATCH_SIZE"],
                                      outdict['learning_rate'],
                                      outdict['Conv2d_1_1'],
                                      outdict['Conv2d_1_2'],
                                      outdict['Conv2d_2_1'],
                                      outdict['Conv2d_2_2'],
                                      loss_test,
                                      acc_test]])
        np.save("log.npy", np.concatenate((old_log, new_log), axis=0))
    return loss_test

space = {"BATCH_SIZE": hp.randint("BATCH_SIZE", 10), # 每一个batch的参数设置，50至150
         "learning_rate": hp.randint("learning_rate", 10), # 学习率的设置，0.05至0.15
         "Conv2d_1_1": hp.randint("Conv2d_1_1", 6), # 第一个卷积块第一层卷积滤波器数量 16至64
         "Conv2d_1_2": hp.randint("Conv2d_1_2", 6), # 第一个卷积块第二层卷积滤波器数量 16至64
         "Conv2d_2_1": hp.randint("Conv2d_2_1", 6), # 第二个卷积块第一层卷积滤波器数量 32至128
         "Conv2d_2_2": hp.randint("Conv2d_2_2", 6), # 第二个卷积块第二层卷积滤波器数量 32至128
        }
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(func, space, algo=algo, max_evals=50)#max_evals表示想要训练的最大模型数量，越大越容易找到最优解

print(best)

logdata = np.load("newlog.npy")

plt.plot(logdata[:, 0], label='BATCH_SIZE curve', c='b')
plt.legend()
plt.title(r'BATCH_SIZE curve')
plt.xlabel('Number of iterations')
plt.ylabel('BATCH_SIZE')  # 设定子图212的Y轴lable为amplitude
plt.savefig(r'BATCH_SIZE curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
plt.show()
plt.close('all')  # 关闭绘图对象，释放绘图资源

plt.plot(logdata[:, 1], label='Learning rate curve', c='b')
plt.legend()
plt.title(r'Learning rate curve')
plt.xlabel('Number of iterations')
plt.ylabel('Learning rate')  # 设定子图212的Y轴lable为amplitude
plt.savefig(r'Learning rate curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
plt.show()
plt.close('all')  # 关闭绘图对象，释放绘图资源

plt.plot(logdata[:, 2], label='conv_1_1', c='r')
plt.plot(logdata[:, 3], label='conv_1_2', c='b', linestyle='--')
plt.legend()
plt.title(r'Convolution number in block 1 curve')
plt.xlabel('Number of iterations')
plt.ylabel('Conv number')  # 设定子图212的Y轴lable为amplitude
plt.savefig(r'Convolution number in block 1 curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
plt.show()
plt.close('all')  # 关闭绘图对象，释放绘图资源

plt.plot(logdata[:, 4], label='conv_2_1', c='r')
plt.plot(logdata[:, 5], label='conv_2_2', c='b', linestyle='--')
plt.legend()
plt.title(r'Convolution number in block 2 curve')
plt.xlabel('Number of iterations')
plt.ylabel('Conv number')  # 设定子图212的Y轴lable为amplitude
plt.savefig(r'Convolution number in block 2 curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
plt.show()
plt.close('all')  # 关闭绘图对象，释放绘图资源

plt.plot(logdata[:, 6], c='b')
plt.title(r'Loss curve')
plt.xlabel('Number of iterations')
plt.ylabel('Loss')  # 设定子图212的Y轴lable为amplitude
plt.savefig(r'Loss curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
plt.show()
plt.close('all')  # 关闭绘图对象，释放绘图资源

plt.plot(logdata[:, 7], c='b')
plt.title(r'Accuracy curve')
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')  # 设定子图212的Y轴lable为amplitude
plt.savefig(r'Accuracy curve.png')  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
plt.show()
plt.close('all')  # 关闭绘图对象，释放绘图资源

print('ok')