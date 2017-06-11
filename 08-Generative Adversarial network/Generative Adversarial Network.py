__author__ = 'ShelockLiao'

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

batch_size = 128
num_epoch = 1000
z_dimension = 100  # noise dimension
learning_rate = 0.0003

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mnist = datasets.MNIST('./data', transform=img_transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True,
                        num_workers=4)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # batch, 32, 24, 24
            nn.ReLU(True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 12, 12
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),  # batch, 64, 8, 8
            nn.ReLU(True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 4, 4
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3249=1x57x57
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            # batch, 50, 57, 57
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=2),
            # batch, 25, 28, 28
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 1, stride=1),  # batch, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 57, 57)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


d = discriminator().cuda()  # discriminator model
g = generator(z_dimension, 3249).cuda()  # generator model

criterion = nn.BCELoss()  # binary cross entropy

d_optimizer = torch.optim.Adam(d.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(g.parameters(), lr=learning_rate)
# x = Variable(torch.randn(1, 100)).cuda()
# x = g(x)
# y = x.cpu().data.unsqueeze(0).numpy()
# y = y.reshape(28, 28)
# plt.imshow(y, cmap='Greys')
# plt.show()

# train
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # =================train discriminator
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # compute loss of real_img
        real_out = d(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = g(z)
        fake_out = d(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = g(z)
        output = d(fake_img)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if (i+1) % 500 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}'
                  'D real: {:.6f}, D fake: {:.6f}'
                  .format(epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                          real_scores.data.mean(), fake_scores.data.mean()))
