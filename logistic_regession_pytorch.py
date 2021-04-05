# 参考：https://blog.csdn.net/out_of_memory_error/article/details/81275651
# https://github.com/rexrex9/FM_recommendation/blob/main/logistic_regression_recommendation.py
import torch
from torch.autograd import Variable
import readData
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset 


epochs = 10
batchSize = 1000
features = 42

def sigmoid(x):
    return 1/(1+torch.exp(-x))

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.w0 = torch.rand(1,1, requires_grad=True)
        self.w = torch.rand(features, 1, requires_grad=True)
        self.params = [self.w, self.w0]

    def forward(self, x):
        # print(x.size())
        # print(self.w.size())
        # x = torch.unsqueeze(x, 2)
        a = torch.matmul(x, self.w) + self.w0
        b = sigmoid(a)
        return b





def dataIter(batch_size,trainX, trainY):
    # dataset = zip(trainX, trainY)
    trainX = torch.Tensor(trainX) # 这里必须是tensor
    trainY = torch.Tensor(trainY)
    dataset = TensorDataset(trainX, trainY)
    train_data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return train_data_iter

def train( trainX, trainY):
    train_data_iter=dataIter(batchSize,trainX,trainY)
    lenTrainY=len(trainY)
    logistic_model = LogisticRegression()
    if torch.cuda.is_available():
        logistic_model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logistic_model.params, lr=1e-3, momentum=0.9)
    for e in range(epochs):
        total_loss = 0
        for x,y in train_data_iter:
            # print(x, y)
            # print(x.size())
            # print(y.size())
            y_hat=logistic_model.forward(x)
            loss = criterion(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += torch.sum(loss)
        print("Epoch %d, average loss:%f" % (e, total_loss / lenTrainY))
    return logistic_model


if __name__ == '__main__':
    trainX, trainY, testX, testY = readData.read_data()
    net=train(trainX,trainY)