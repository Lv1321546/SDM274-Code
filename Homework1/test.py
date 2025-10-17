import numpy as np
import matplotlib.pyplot as plt

# print("Linear Regression Model")

class LinearRegression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.w0 = np.random.normal(0,0.01)
        self.w1 = np.random.normal(0,0.01)
        # self.w0 = 10
        # self.w1 = 1

    def compute(self, input):
        output = self.w0 + self.w1*input
        return output
    
    def loss_function(self, x_data=None):
        MSE = 0
        self.N = self.X_train.size
        x_data = self.X_train if x_data is None else x_data

        for i in range(0, self.N):
            xi = x_data[i]
            yi_model = self.compute(xi)
            yi_true = self.y_train[i]
            SE = (yi_true-yi_model)**2
            MSE += SE/self.N
            # print(xi, yi_model, yi_true)

        return MSE
    
    def plot(self, method='SGD', normalization='No'):
        y_line = self.w0 + self.w1 * self.X_train

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_train, self.y_train, color = 'blue', label = 'Training_data')
        plt.plot(self.X_train, y_line, color='red', label='Regression Result', linewidth=3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, 100)
        plt.ylim(0, 120)
        plt.title(f"{method} with {normalization} Normalization")
        plt.legend()
        
        # 保存图片
        filename = f"Homework1/plots/{method}_{normalization}_normalization.png"
        plt.savefig(filename)
        plt.close()

    def MinMax_Normalzation(self, Array):
        self.Min = Array.min()
        self.Max = Array.max()
        self.Diff = self.Max - self.Min
        Brray = (Array - self.Min)/self.Diff
        return Brray
    
    def Mean_Normalzation(self, Array):
        self.Mean = np.mean(Array, axis=0)
        self.Sigma = np.std(Array, axis=0)
        Brray = (Array - self.Mean) / self.Sigma
        return Brray
    
    def plot_loss(self, loss_history, method='SGD', normalization='No'):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(f"Loss Curve - {method} with {normalization} Normalization")
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        filename = f"Homework1/plots/{method}_{normalization}_loss.png"
        plt.savefig(filename)
        plt.close()
    
    # def inverse_MinMax_Normalzation(self, Brray):
    #     Min = Brray.min
    #     Max = Brray.max
    #     Diff = Max - Min
    #     Crray = Brray*Diff+Min
    #     return Crray
    def OLS(self):
        X = np.c_[np.ones((len(self.X_train), 1)), self.X_train.reshape(-1, 1)]
        y = self.y_train.reshape(-1, 1)

        theta, *_ = np.linalg.lstsq(X, y, rcond=None) 

        self.w0 = float(theta[0, 0])
        self.w1 = float(theta[1, 0])
        print(f"[最小二乘法] w0 = {self.w0:.3f}, w1 = {self.w1:.3f}")

    def GD(self, lr = 5e-4, epoch = 5000, Normalzation = 'No', Method = 'SGD', batch_size = 25):

        self.w0 = np.random.normal(0,0.01)
        self.w1 = np.random.normal(0,0.01)
        
        # 初始化loss记录列表
        loss_history = []
        
        if Normalzation == 'MinMax':
            X_train_norm = self.MinMax_Normalzation(self.X_train)
        elif Normalzation == 'Mean':
            X_train_norm = self.Mean_Normalzation(self.X_train)
        else:
            X_train_norm = self.X_train

        for k in range(epoch):
            if Method == 'SGD':
                index = np.array(range(0, self.N))
                np.random.shuffle(index)
                for j in range(0, self.N):
                    i = index[j] # 乱序
                    x_i = float(X_train_norm[i])
                    yi_model = self.compute(x_i)
                    # print(j, i, x_i, yi_model, self.y_train[i] )
                    self.w0 += lr*(self.y_train[i] - yi_model)
                    self.w1 += lr*(self.y_train[i] - yi_model)*x_i
            elif Method == 'BGD':
                x = np.asarray(X_train_norm, dtype=float).ravel()
                y = np.asarray(self.y_train, dtype=float).ravel()

                yi_hat = self.w0 + self.w1*x
                error = y - yi_hat
                self.w0 += lr*error.mean()
                self.w1 += lr*(error*x).mean()
                # print(error.mean())
                
            elif Method == 'MBGD':
                x = np.asarray(X_train_norm, dtype=float).ravel()
                y = np.asarray(self.y_train, dtype=float).ravel()
                index = np.array(range(0, self.N))
                np.random.shuffle(index)
                for j in range(0, self.N, batch_size):
                    i = index[j:j+batch_size]
                    x_i = x[i]
                    yi_model = self.w0 + self.w1*x
                    yi_model = yi_model[i]
                    yi_train = y[i]

                    error = yi_train-yi_model
                    self.w0 += lr * error.mean()
                    self.w1 += lr * (error * x_i).mean()

                 
            else:
                return

            # 记录当前loss，使用正在训练的数据计算loss
            current_loss = self.loss_function(x_data=X_train_norm)
            loss_history.append(current_loss)

        if Normalzation == 'MinMax':
            self.w0 = self.w0 - self.w1*self.Min/self.Diff
            self.w1 = self.w1/self.Diff
            print(f"[Min-Max归一化] w0 = {float(self.w0):.3f}, w1 = {float(self.w1):.3f}")
        elif Normalzation == 'Mean':
            self.w0 = self.w0 - self.w1 * self.Mean / self.Sigma
            self.w1 = self.w1 / self.Sigma       
            print(f"[Mean归一化]w0 = {float(self.w0):.3f}, w1 = {float(self.w1):.3f}") 
        else:
            print(f"[无归一化]w0 = {float(self.w0):.3f}, w1 = {float(self.w1):.3f}")
            
        return loss_history

        

def main():
    X_train = np.arange(100).reshape(100,1)  # from 0 to 99, matrix 100*1
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)  # y = x + 10 + error
    y_train = y_train.reshape(-1)  # convert to 1-D array

    LR_instant = LinearRegression(X_train, y_train)
    loss = LR_instant.loss_function()

    # print(loss)
    # LR_instant.SGD()
    # LR_instant.plot()
    # loss = LR_instant.loss_function()
    # print(loss)

    ## 最小二乘法
    LR_instant.OLS()

    print("SGD结果")

    ## SGD无归一化
    loss_history = LR_instant.GD(lr = 5e-5, epoch = 5000, Method='SGD')
    LR_instant.plot(method='SGD', normalization='No')
    LR_instant.plot_loss(loss_history, method='SGD', normalization='No')

    # ## SGD Min-Max归一化
    # loss_history = LR_instant.GD(lr = 5e-4, epoch = 5000, Normalzation= 'MinMax', Method='SGD')
    # LR_instant.plot(method='SGD', normalization='MinMax')
    # LR_instant.plot_loss(loss_history, method='SGD', normalization='MinMax')

    # ## SGD Mean归一化
    # loss_history = LR_instant.GD(lr = 5e-4, epoch = 5000, Normalzation= 'Mean', Method='SGD')
    # LR_instant.plot(method='SGD', normalization='Mean')
    # LR_instant.plot_loss(loss_history, method='SGD', normalization='Mean')

    print("BGD结果")

    ## BGD无归一化
    loss_history = LR_instant.GD(lr = 5e-4, epoch = 50000, Method='BGD')
    LR_instant.plot(method='BGD', normalization='No')
    LR_instant.plot_loss(loss_history, method='BGD', normalization='No')

    # ## BGD Min-Max归一化
    # loss_history = LR_instant.GD(lr = 5e-3, epoch = 50000, Normalzation= 'MinMax', Method='BGD')
    # LR_instant.plot(method='BGD', normalization='MinMax')
    # LR_instant.plot_loss(loss_history, method='BGD', normalization='MinMax')
    
    # ## BGD Mean归一化
    # loss_history = LR_instant.GD(lr = 5e-4, epoch = 50000, Normalzation= 'Mean', Method='BGD')
    # LR_instant.plot(method='BGD', normalization='Mean')
    # LR_instant.plot_loss(loss_history, method='BGD', normalization='Mean')

    print("MBGD结果")

    ## MBGD无归一化
    loss_history = LR_instant.GD(lr = 5e-4, epoch = 50000, Method='MBGD')
    LR_instant.plot(method='MBGD', normalization='No')
    LR_instant.plot_loss(loss_history, method='MBGD', normalization='No')

    # ## MBGD Min-Max归一化
    # loss_history = LR_instant.GD(lr = 5e-3, epoch = 5000, Normalzation= 'MinMax', Method='MBGD')
    # LR_instant.plot(method='MBGD', normalization='MinMax')
    # LR_instant.plot_loss(loss_history, method='MBGD', normalization='MinMax')
    
    # ## MBGD Mean归一化
    # loss_history = LR_instant.GD(lr = 5e-4, epoch = 5000, Normalzation= 'Mean', Method='MBGD')
    # LR_instant.plot(method='MBGD', normalization='Mean')
    # LR_instant.plot_loss(loss_history, method='MBGD', normalization='Mean')

if __name__ == "__main__":  
    main()


