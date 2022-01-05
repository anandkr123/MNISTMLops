import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Network


# Hyper-parameters

epochs = 1
train_batch_size = 100


def train(model, train_loader, device, optimizer):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

    #         if batch % 100 == 0:
    #             print(f"Train Epoch: {epoch} [{batch * len(data)}/{len(train_loader.dataset)}({(100. * batch / len(train_loader)):.0f}%)] \tLoss: {loss.item():.6f}")
    with open("results.txt", 'w') as outfile:
        outfile.write("Epoch: %2.1f%%\n" % epoch)
        outfile.write("Train loss: %2.1f%%\n" % loss.item())

    print(f"\n Train Epoch: {epoch} \tLoss: {loss.item():.6f}")



train_set = torchvision.datasets.FashionMNIST('./data/FashionMNIST',
                                              train = True,
                                              download = True,
                                             transform = transforms.Compose([
                                                 transforms.ToTensor()
                                             ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=0)


for epoch in range(1, epochs + 1):
    train(model, train_loader, device, optimizer)
    # test(model, test_loader, device)
