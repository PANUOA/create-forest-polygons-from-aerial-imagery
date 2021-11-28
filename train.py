from model.unet import unet
from utils.dataset import z15_Loader
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment')


def train_net(net, device, data_path, epochs=1, batch_size=8, lr=0.00001):
    lostlist = []
    round = 0
    z15_dataset = z15_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=z15_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            round = round + 1
            image = np.transpose(image, (0, 3, 1, 2))
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()
            lostlist.append(loss.item())
            writer.add_scalar("Loss/round", loss.item(), round)
            writer.add_graph(net, [image])
            writer.flush()
            writer.close()
            plt.plot(lostlist)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = unet(n_channels=3, n_classes=1)
    net.to(device=device)
    data_path = "data/train/"
    train_net(net, device, data_path)
    plt.show()
