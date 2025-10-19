# 葡萄酒识别数据

该数据集是对在意大利同一地区种植但来自三个不同品种的葡萄酒进行化学分析的结果。该分析确定了在三种类型的葡萄酒中发现的13种成分的含量。

## 数据集信息

- 实例数：178
- 属性数：13个连续属性

## 属性信息

属性如下：
1. 酒精
2. 苹果酸
3. 灰分
4. 灰分的碱度
5. 镁
6. 总酚
7. 黄酮类化合物
8. 非黄酮类酚
9. 原花青素
10. 颜色强度
11. 色调
12. 稀释葡萄酒的OD280/OD315
13. 脯氨酸

`wine.data` 文件中的第一列代表类别标识符（1-3）。

## 类别分布

- 类别1：59个实例
- 类别2：71个实例
- 类别3：48个实例

## 作业要求

1.  **数据准备:**
    *   `wine.data` 数据集包含178个数据点，每个数据点代表一个葡萄酒样本，包含一个类别标签和13个特征（化学指标，如酒精含量和苹果酸浓度）。更详细的信息请参见 `wine.names` 文件。
    *   从数据集中删除一个类别的葡萄酒样本，保留另外两个类别以生成新的数据集。
    *   将数据集按7:3的比例分割为训练集和测试集。

2.  **模型与训练:**
    *   使用逻辑回归模型和交叉熵损失函数来解决葡萄酒的二元分类问题。
    *   编写小批量更新和随机更新的代码来训练模型。

3.  **评估:**
    *   使用训练好的模型对测试集进行预测。
    *   编写代码，使用准确率（Accuracy）、召回率（Recall）、精确率（Precision）和F1分数（F1 score）来评估模型的分类性能。

The wine.data dataset contains a total of 178 data points, each representing a wine sample with a class label and 13 features, which are chemical indicators such as alcohol content and malic acid concentration. More detailed information can be found the file titled wine.names.

Remove one class of wine samples from the dataset, retaining the other two classes to generate a new dataset. 

Consider the logistic regression model and cross-entropy loss function to address the binary classification problem of wine.

1.Write code to split the dataset into a training set and a test set in the ratio of (0.7, 0.3).
2.Write codes for Mini-batch update and Stochastic update to train the model.
3.Use the trained model to make predictions on the test set, and write code to evaluate the model's classification performance using Accuracy, recall, precision, and F1 score.