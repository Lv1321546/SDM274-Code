import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # 只用来导入MNIST数据集

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_test, y_test

def data_preprocess():
    np.random.seed(42)

    x_train_orig, y_train_orig, x_test_orig, y_test_orig = load_data()
    X = np.concatenate([x_train_orig, x_test_orig], axis=0).astype(np.float32)
    Y = np.concatenate([y_train_orig, y_test_orig], axis=0)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(0,10):
        index = np.where(Y==i)[0]
        num = len(index)
        np.random.shuffle(index)

        x_train.append(X[index[:int(num*0.8)]])
        x_test.append(X[index[int(num*0.8):]])
        y_train.append(Y[index[:int(num*0.8)]])    
        y_test.append(Y[index[int(num*0.8):]])

    X_train_raw = np.concatenate(x_train, axis=0)
    y_train_raw = np.concatenate(y_train, axis=0)
    X_test_raw = np.concatenate(x_test, axis=0)
    y_test_raw = np.concatenate(y_test, axis=0)

    
    X_mean = np.mean(X_train_raw, axis=0)
    X_range = np.ptp(X_train_raw, axis=0) # Max - Min
    X_range[X_range == 0] = 1 
    
    X_train_norm = (X_train_raw - X_mean) / X_range
    X_test_norm = (X_test_raw - X_mean) / X_range 

    index2 = np.array(range(X_train_norm.shape[0]))
    np.random.shuffle(index2)
    X_train_result = X_train_norm[index2]
    y_train_result = y_train_raw[index2]   

    index3 = np.array(range(X_test_norm.shape[0]))
    np.random.shuffle(index3)  
    X_test_result = X_test_norm[index3]
    y_test_result = y_test_raw[index3]

    # 在返回之前进行one-hot编码
    def to_one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]
    
    y_train_one_hot = to_one_hot(y_train_result)
    y_test_one_hot = to_one_hot(y_test_result)
    
    print(y_test_result[:100])
    return X_train_result, y_train_one_hot, X_test_result, y_test_one_hot

def visualize_samples_normalized(X_norm, Y):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        indices = np.where(Y == i)[0]
        if len(indices) > 0:
            sample_index = indices[100]
            img_display = np.maximum(0, X_norm[sample_index]).reshape(28, 28)
            plt.imshow(img_display, cmap='gray')
            plt.title(f'Label: {i}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


class LogisticRegression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.W = np.zeros((785,10))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def predict_function(self, X):
        X_bar = np.vstack([np.ones(X.shape[0]), X.T]) #  添加偏置 785*n
        Z = np.dot(self.W.T, X_bar) # 10*n
        Y = np.exp(Z) / np.sum(np.exp(Z), axis=0)  # Softmax函数
        return X_bar, Y.T  # n*10
    
    def gredient(self, X, y_true, alpha=0.001):
        X_bar, y_pred = self.predict_function(X)  # n*10
        Num = X.shape[0]
        # y_true 已经是one-hot编码，可以直接相减
        gredient = 1/Num * X_bar.dot(y_pred - y_true) + alpha*self.W  # 785*10
        return gredient
            
    def update(self, epoch=5000, lr=0.001, batch_size=32): 
        train_num = self.X_train.shape[0]
        for k in range(epoch): # 重复训练epoch次
            index = np.array(range(0, train_num))
            np.random.shuffle(index)
            for j in range(0, train_num, batch_size):
                i = index[j:j+batch_size]
                x_i = self.X_train[i]  # 使用self.X_train
                y_i = self.y_train[i]  # 使用self.y_train
                gred = self.gredient(x_i, y_i)
                self.W -= lr * gred
            
            # 可以添加每隔一定epoch打印训练进度
            if k % 100 == 0:
                train_acc = self.evaluate(is_test=False)
                print(f"Epoch {k}, Training Accuracy: {train_acc:.2%}")

    def evaluate(self, is_test=True):  # 添加参数来选择评估训练集还是测试集
        if is_test:
            X = self.X_test
            y = self.y_test
        else:
            X = self.X_train
            y = self.y_train
            
        _, y_pred = self.predict_function(X)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)  # 从one-hot转回标签
        accuracy = np.mean(y_pred_labels == y_true_labels)
        
        if is_test:  # 只在评估测试集时打印
            print(f"Test Accuracy: {accuracy:.2%}")
        return accuracy

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_preprocess()
    # visualize_samples_normalized(X_train, y_train)
    
    # 创建模型实例
    logisticRegression = LogisticRegression(X_train, y_train, X_test, y_test)
    
    # 训练模型
    logisticRegression.update(epoch=5000, lr=0.0005, batch_size=32)
    
    # 评估模型
    logisticRegression.evaluate()  # 在测试集上评估
