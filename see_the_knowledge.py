import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets,transforms

# 不知道MNIST数据集路径对不对
test_loader_bs1 = torch.utils.data.DataLoader(
    datasets.MNIST(root="dataset/", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=True)

def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)

def function_see_the_knowledge(test_net, test_loader_bs1=test_loader_bs1):
    test_net.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader_bs1))
        data, target = data.to('cuda'), target.to('cuda')
        output = test_net(data)
    test_x = data.cpu().numpy()
    y_out = output.cpu().numpy()
    y_out = y_out[0, ::]
    print('Output (NO softmax):', y_out)

    plt.subplot(3, 1, 1)
    plt.imshow(test_x[0, 0, ::])

    plt.subplot(3, 1, 2)
    plt.bar(list(range(10)), softmax_t(y_out, 1), width=0.3)

    plt.subplot(3, 1, 3)
    plt.bar(list(range(10)), softmax_t(y_out, 10), width=0.3)
    plt.show()