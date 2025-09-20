作业1：
    用 Python + NumPy 实现一个 LinearRegression 类（线性回归）。
    使用 MSE（均方误差）作为损失函数。
    
    在类中实现 三种更新方法：
        BGD（Batch Gradient Descent，批量梯度下降）
        SGD（Stochastic Gradient Descent，随机梯度下降）
            在每个 epoch（遍历一遍训练集）开始前，先随机打乱样本顺序。
            然后按顺序一个个取样本做 SGD 更新。
        MBGD（Mini-batch Gradient Descent，小批量梯度下降）

    在类中加入两种归一化选项：
        min-max normalization（最小-最大归一化）
        mean normalization（均值归一化/标准化的一种变体；按要求实现你课程指定的公式）