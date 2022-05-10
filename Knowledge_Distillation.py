'''在本地调试代码，在colab训练'''

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt


# 设置随机种子
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark=True # 使用cuDNN加速卷积运算

# 统计
A, B, C = [],[],[]

# 定义教师网络
from teacher import TeacherNet
tea_net = TeacherNet()
tea_net.to(device)

# 准备数据集并加载
train_dataset = torchvision.datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root="dataset/",train=False,transform=transforms.ToTensor(),download=True)
train_loder = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loder  = DataLoader(dataset=test_dataset, batch_size=32,shuffle=False)

criterion_tea = nn.CrossEntropyLoss() # 设置使用交叉熵损失函数
criterion_tea = criterion_tea.to(device)
optimizer_tea = torch.optim.Adam(tea_net.parameters(),lr=1e-4) # 使用Adam优化器，学习率为lr=1e-4


# 训练并测试教师网络
for epoch in range(10):
    tea_net.train()

    for data, targets in tqdm(train_loder):
        data, targets = data.to(device),targets.to(device)
        preds = tea_net(data)
        loss = criterion_tea(preds, targets)
        optimizer_tea.zero_grad()
        loss.backward()
        optimizer_tea.step()

    tea_net.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loder:
            x, y = x.to(device), y.to(device)
            preds = tea_net(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
    A.append([loss,acc])
    print(("Train_TeaNet_Epoch:{}\t Accuracy:{:4f}").format(epoch + 1, acc))

# 是否要看教师网络训练出来的知识
see = True
if see:
    from see_the_knowledge import function_see_the_knowledge
    function_see_the_knowledge(tea_net)

# 定义学生网络
from student import StudentNet
stu_net = StudentNet()

# 单独训练一下学生网络
criterion_stu = nn.CrossEntropyLoss()
optimizer_stu = torch.optim.Adam(stu_net.parameters(),lr=1e-4)
for epoch in range(10):
    stu_net.train()

    for data,targets in tqdm(train_loder):
        data, targets = data.to(device),targets.to(device)
        preds = stu_net(data)
        loss = criterion_stu(preds,targets)
        optimizer_stu.zero_grad() # 把梯度置为0
        loss.backward()
        optimizer_stu.step()

    with torch.no_grad():
        for x,y in  test_loder:
            x, y = x.to(device), y.to(device)
            preds = stu_net(x)
            predictions = preds.max(1).indices
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            acc = (num_correct / num_samples).item()
    B.append([loss, acc])
    print(("when_WithouKD_Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))

# 准备好训练好的教师模型
tea_net = tea_net.to(device)
tea_net.eavl()

# 准备新的学生模型
new_stu_net = StudentNet()
new_stu_net.to(device)
temp = 5 # 蒸馏温度

# hard_loss及其在总损失的占比
hard_loss = nn.CrossEntropyLoss()
hard_loss = hard_loss.to(device)
alpha = 0.3

# soft_loss and optimizer
soft_loss = nn.KLDivLoss()
soft_loss = soft_loss.to(device)
optimizer_kd = torch.optim.Adam(new_stu_net.parameters(),lr=1e-4)

# 用知识蒸馏训练学生网络
for epoch in range(10):
    new_stu_net.tarin()
    for data,targets in tqdm(train_loder):
        data, targets = data.to(device),targets.to(device)
        # 教师模型预测
        with torch.no_grad(): # 保证教师模型的梯度不会更新
            teacher_preds = tea_net(data)

        # 学生模型预测
        student_preds = new_stu_net(data)

        student_loss = hard_loss(student_preds,targets)

        # 计算蒸馏后的预测结果及soft_loss
        distillation_loss = soft_loss(F.softmax(student_preds/temp, dim=1), F.softmax(teacher_preds/temp, dim=1))

        # 将 hard_loss 和 soft_loss 加权求和
        loss = alpha * student_loss + (1-alpha) * distillation_loss

        # 反向传播,优化权重
        optimizer_kd.zero_grad()
        loss.backward()
        optimizer_kd.step()

    # 测试集上评估性能
    new_stu_net.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in test_loder:
            x, y = x.to(device), y.to(device)
            preds = new_stu_net(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()
    C.append([loss, acc])
    print(("when_WithKD_Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))

draw = True
if draw:
    # 画一下三种情况的训练损失与测试准确率
    x = list(range(1,11))

    plt.subplot(2, 1, 1)
    plt.plot(x, [A[i][1] for i in range(10)], label='teacher')
    plt.plot(x, [B[i][1] for i in range(10)], label='student withOUT KD')
    plt.plot(x, [C[i][1] for i in range(10)], label='student with KD')
    plt.title('Test Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, [A[i][0] for i in range(10)], label='teacher')
    plt.plot(x, [B[i][0] for i in range(10)], label='student withOUT KD')
    plt.plot(x, [C[i][0] for i in range(10)], label='student with KD')

    plt.title('Test Accuracy')
    plt.legend()