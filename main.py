"""
模块是一个包含所有你定义的函数和变量的文件，其后缀名是.py。
模块可以被别的程序引入，以使用该模块中的函数等功能。这也是使用 python 标准库的方法。
用import来导入模块
"""
from __future__ import print_function
# 这样的做法的作用就是将新版本的特性引进当前版本中，也就是说我们可以在当前版本使用新版本的一些特性
# argparse是python标准库里面用来处理命令行参数的库
import argparse
import torch
# import语句 使用自己想要的名字替换模块的原始名称
import torch.nn as nn
import torch.nn.functional as F
# 此包用于内含更新权值的各种方法 类如随机梯度下降SGD
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# nn.Module : Base class for all neural network modules
class Net(nn.Module):
    # 基类定义在另一个模块中时的继承写法
    def __init__(self):
        # 调用父类构造函数
        super(Net, self).__init__()
        # 输入通道为1  输出通道为32 卷积核大小为3  步长为1 padding为0
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 防止过拟合  按照0.25概率将网络参数置零
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # 变换形状  适应全连接层输入
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # mathematically equivalent to log(softmax(x))
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # 遍历数据集
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        '''
        forward + backward + optimize
        '''
        # 前向通过网络
        output = model(data)
        # 用nll_loss计算损失
        loss = F.nll_loss(output, target)
        # 梯度反向传播
        loss.backward()
        # 更新参数/权值
        optimizer.step()
        # 每10个batch打印一次信息  因为args.log_interval = 10
        if batch_idx % args.log_interval == 0:
            # len(train_loader)的值等于batch个数，
            # len(train_loader.dataset)是数据集的大小
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 不跟踪梯度   requires_grad=false
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # Computes element-wise equality
            correct += pred.eq(target.view_as(pred)).sum().item()
            # View this tensor as the same size as other. self.view_as(other) is equivalent to self.view(other.size()).
            # tensor 变个形状

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # The parse_args() method supports several ways of specifying the value of an option
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # Sets the seed for generating random numbers
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    '''   
    还可以使用分开的更明了的写法  初学这个版本看起来更明了一些  熟练了应该都是按照上面的写法
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size,
                                          shuffle=True, **kwargs)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                         shuffle=True, **kwargs)
    
    '''

    model = Net().to(device)

    # 使用Adadelta方法进行权值更新
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    # 保存训练模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


# 每个模块都有一个__name__属性，
# 当其值是'__main__'时，表明该模块自身在运行，否则是被引入。
# 用if __name__ == '__main__'来判断是否是在直接运行该.py文件
if __name__ == '__main__':
    main()
