import numpy as np

X_train = np.arange(100).reshape(100,1)  # from 0 to 99, matrix 100*1
a, b = 1, 10
y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)  # y = x + 10 + error
y_train = y_train.reshape(-1)  # convert to 1-D array

# print(X_train)
# # print(y_train)
# print(X_train.size)
print(X_train[1])
# # print(X_train[1]+1)

# # for i in range(0, 10):
# #     print(i)

print(y_train[0])

# Array = np.array(range(0, 100))
# print(Array)
