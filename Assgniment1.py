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
        Num = self.X_train.size

        for i in range(0, self.X_train.size):
            xi = self.X_train[i]
            yi_model = self.compute(xi)
            yi_true = self.y_train[i]
            SE = (yi_true-yi_model)**2
            MSE += SE/Num
            # print(xi, yi_model, yi_true)

        return MSE
    
    def plot(self):
        y_line = self.w0 + self.w1 * X_train

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
        # print(self.Min)
        self.Diff = self.Max - self.Min
        Brray = (Array - self.Min)/self.Diff
        return Brray
    
    # def inverse_MinMax_Normalzation(self, Brray):
    #     Min = Brray.min
    #     Max = Brray.max
    #     Diff = Max - Min
    #     Crray = Brray*Diff+Min
    #     return Crray

    def SGD(self, lr = 5e-4, epoch = 5000, Normalzation = 'MinMax'):
        if Normalzation == 'MinMax':
            X_train_norm = self.MinMax_Normalzation(self.X_train)
        elif Normalzation == 'Mean':
            pass
        else:
            X_train_norm = self.X_train

        for k in range(epoch):
            index = np.array(range(0, 100))
            np.random.shuffle(index)
            for j in range(0, 100):
                i = index[j] # 乱序
                x_i = float(X_train_norm[i])
                yi_model = self.compute(x_i)
                # print(j, i, x_i, yi_model, self.y_train[i] )
                self.w0 += lr*(self.y_train[i] - yi_model)
                self.w1 += lr*(self.y_train[i] - yi_model)*x_i

        self.w0 = self.w0 - self.w1*self.Min/self.Diff
        self.w1 = self.w1/self.Diff
        print(self.w0, self.w1)

       


X_train = np.arange(100).reshape(100,1)  # from 0 to 99, matrix 100*1
a, b = 1, 10
y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)  # y = x + 10 + error
y_train = y_train.reshape(-1)  # convert to 1-D array

LR_instant = LinearRegression(X_train, y_train)
loss = LR_instant.loss_function()

print(loss)
LR_instant.SGD()
LR_instant.plot()
loss = LR_instant.loss_function()
print(loss)
