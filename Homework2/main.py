import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinaryClassification:
    def __init__(self):
        self.feautre_num = 13
        # self.lr = 0
        self.W = np.random.randn(self.feautre_num+1)*0.05
        # self.loss = 0
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        sigmoid_z = 1/(1+np.exp(-z))
        return sigmoid_z
    
    def loss(self, y_true, y_perd):
        loss = -y_true*np.log(y_perd + 1e-5)-(1-y_true)*np.log(1-y_perd + 1e-5)
        return loss

    def pred_function(self, X): # X的大小应该是 n*14（第一行是1）
        Y_pred = self.sigmoid(np.dot(X, self.W)) # 计算y数组，结果是n*1
        return Y_pred
    
    def gredient(self, X, y_true):
        gredient = -np.dot(X.T, (y_true - self.pred_function(X))) # 结果是14*1
        return gredient
    
    def data_process(self):
        data = pd.read_csv('Homework2\wine.data', header=None)
        X_raw = data.iloc[:, 1:].values # feature数组，178*13
        y_raw = data.iloc[:, 0].values # label数组，178*1
        
        class_remove = 3
        X_new = X_raw[data[0] != class_remove] # n*13
        y_new = y_raw[data[0] != class_remove] # n*1
        data_num = X_new.shape[0]
        y_new = np.where(y_new == 1, 0, 1)

        # np.random.seed(42)
        
        total_samples = X_new.shape[0]
        train_size = int(total_samples * 0.7)
        
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X_new = np.column_stack([np.ones(X_new.shape[0]), X_new])  # 在最左边添加一列1

        X_train = X_new[train_indices]
        X_test = X_new[test_indices]
        y_train = y_new[train_indices]
        y_test = y_new[test_indices]
            
        return X_train, X_test, y_train, y_test
    
    def update(self, type, X_train, y_train, epoch = 5000, lr = 0.0001):
        train_num = X_train.shape[0]
        for k in range(epoch): # 重复训练epoch次
            if type == "Mini-batch":
                batch_size =20
                index = np.array(range(0, train_num))
                np.random.shuffle(index)
                for j in range(0, train_num, batch_size):
                    i = index[j:j+batch_size]
                    x_i = X_train[i]
                    y_i = y_train[i]
                    gred = self.gredient(X_train[i], y_train[i])
                    self.W = self.W - lr*gred

            elif type == "Stochastic":
                index = np.array(range(0, train_num))
                np.random.shuffle(index) # 打乱顺序
                for j in range(train_num): # 乱序遍历训练集
                    i = index[j]
                    # Xi = float(X_train[i])
                    gred = self.gredient(X_train[i], y_train[i])
                    self.W = self.W - lr*gred

            else:
                pass

    def validation(self, X_test, y_test):
        test_num = X_test.shape[0]
        # for i in range(0,test_num):
        y_pred = self.pred_function(X_test)
        accurate_num = np.sum(np.abs(y_pred - y_test)<0.5)
            
        TP = np.sum((y_pred == 1) & (y_test == 1))  # 真正例
        TN = np.sum((y_pred == 0) & (y_test == 0))  # 真负例
        FP = np.sum((y_pred == 1) & (y_test == 0))  # 假正例
        FN = np.sum((y_pred == 0) & (y_test == 1))  # 假负例
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        
        print(f"Metrics: Accuracy={accuracy:.2%}, Precision={precision:.2%}, Recall={recall:.2%}, F1 Score={f1_score:.2%}")

if __name__ == "__main__":
    bc = BinaryClassification()

    X_train, X_test, y_train, y_test = bc.data_process()
    train_num = X_train.shape[0] # 91*13
    test_num = X_test.shape[0] # 39*13
    # print(X_train)
    # print(bc.W.shape)

    # print(bc.W)
    print("Stochastic Gredient Descent:")
    bc.update(type="Stochastic", X_train=X_train, y_train= y_train, lr=0.0001, epoch= 5000)
    bc.validation(X_test, y_test)

    print("Mini-batch Gredient Descent:")
    bc.update(type="Mini-batch", X_train=X_train, y_train= y_train, lr=0.0001, epoch= 5000)
    bc.validation(X_test, y_test)
    

