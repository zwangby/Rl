#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhixiang time:18-3-31


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = x.pow(2) + 0.2*torch.rand(x.size())
# x, y = Variable(x), Variable(y)


n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], y.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdY1Gn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
# zhe ge di fang relu huiyou jieduan ,meiyou fushu bufen
        x = self.predict(x)
        return x


net = Net(2, 10, 2)

print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()


for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
plt.pause(0.1)

plt.ioff()
plt.show()
