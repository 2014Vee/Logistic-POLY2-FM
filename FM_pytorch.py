# 参考：https://blog.csdn.net/out_of_memory_error/article/details/81275651
# https://github.com/rexrex9/FM_recommendation/blob/main/logistic_regression_recommendation.py
# 2014Vee 
# 通过pytorch实现FM算法，主要测试一下git功能
import torch
from torch.autograd import Variable
import readData
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset 
from tqdm import tqdm


epochs = 10
batchSize = 1000
features = 42

def sigmoid(x):
    return 1/(1+torch.exp(-x))

class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()
        self.k = 10  # 设置隐向量的维度
        self.w0 = torch.rand(1,1, requires_grad=True)
        self.w = torch.rand(features, 1, requires_grad=True)
        self.bw = torch.rand(features, self.k, requires_grad=True) # 这里对每个特征都是用隐向量权重表示的
        self.params = [self.w, self.w0, self.bw]

    def __getTwoCross(self, X):
        t = None
        # 因为输入的时是batch的数据，所以需要进行考虑
        batch = 0
        # 加入进度条显示tqdm
        for x in tqdm(X):   
            s=0
            for j1 in range(len(x)):
                for j2 in range(j1+1, len(x)):
                    # print(x[j1])
                    # print(x[j2])
                    # print(torch.mul(self.bw[j1], self.bw[j2]))
                    # 这里被坑了torch.dot乘完是标量，torch.mul乘完还是tensor这俩维度不一样切记
                    tmp = torch.dot(self.bw[j1], self.bw[j2])
                    s += tmp * x[j1] * x[j2]
                    s = torch.tensor([s])
                    # print("s:", s.size())
            if batch == 0:
                t = s
            else:
                # print(t.size())
                # print(s.size())
                t = torch.cat((t, s))
            batch += 1
        # 这里加法部分需要注意，tensor加法运算，size[a,1]+size[b]=size[a,b]
        # print("t:", t.size())#维度不对怎么是10000
        t = torch.unsqueeze(t, 1)
        return t

    def forward(self, x):
        # print(x.size())
        # print(self.w.size())
        # x = torch.unsqueeze(x, 2)
        print(self.__getTwoCross(x).size())
        a = torch.matmul(x, self.w) + self.w0 + self.__getTwoCross(x)
        b = sigmoid(a)
        print(b.size())
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
    FM_model = FM()
    if torch.cuda.is_available():
        FM_model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(FM_model.params, lr=1e-3, momentum=0.9)
    for e in range(epochs):
        total_loss = 0
        for x,y in train_data_iter:
            # print(x, y)
            # print(x.size())
            # print(y.size())
            y_hat=FM_model.forward(x) 
            loss = criterion(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += torch.sum(loss)
        print("Epoch %d, average loss:%f" % (e, total_loss / lenTrainY))
    return FM_model


if __name__ == '__main__':
    trainX, trainY, testX, testY = readData.read_data()
    net=train(trainX,trainY)