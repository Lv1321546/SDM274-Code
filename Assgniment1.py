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
    
    def loss_function(self):
        MSE = 0
        self.N = self.X_train.size

        for i in range(0, self.X_train.size):
            xi = self.X_train[i]
            yi_model = self.compute(xi)
            yi_true = self.y_train[i]
            SE = (yi_true-yi_model)**2
            MSE += SE/self.N
            # print(xi, yi_model, yi_true)

        return MSE
    
    def plot(self):
        y_line = self.w0 + self.w1 * self.X_train

        plt.scatter(self.X_train, self.y_train, color = 'blue', label = 'Training_data')
        plt.plot(self.X_train, y_line, color='red', label='Regression Result', linewidth=3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, 100)
        plt.ylim(0, 120)
        plt.legend()
        plt.show()

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

    def SGD(self, lr = 5e-4, epoch = 5000, Normalzation = 'No'):

        self.w0 = np.random.normal(0,0.01)
        self.w1 = np.random.normal(0,0.01)
    
        if Normalzation == 'MinMax':
            X_train_norm = self.MinMax_Normalzation(self.X_train)
        elif Normalzation == 'Mean':
            X_train_norm = self.Mean_Normalzation(self.X_train)
        else:
            X_train_norm = self.X_train

        for k in range(epoch):
            index = np.array(range(0, self.N))
            np.random.shuffle(index)
            for j in range(0, self.N):
                i = index[j] # 乱序
                x_i = float(X_train_norm[i])
                yi_model = self.compute(x_i)
                # print(j, i, x_i, yi_model, self.y_train[i] )
                self.w0 += lr*(self.y_train[i] - yi_model)
                self.w1 += lr*(self.y_train[i] - yi_model)*x_i

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

    ## SGD无归一化
    LR_instant.SGD(lr = 5e-5, epoch = 5000)
    # LR_instant.plot()

    ## SGD Min-Max归一化
    LR_instant.SGD(lr = 5e-4, epoch = 5000, Normalzation= 'MinMax')
    # LR_instant.plot()

    ## SGD Mean归一化
    LR_instant.SGD(lr = 5e-4, epoch = 5000, Normalzation= 'Mean')
    # LR_instant.plot()

    
if __name__ == "__main__":  
    main()


