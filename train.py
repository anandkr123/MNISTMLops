import json

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Network
import mlflow

# Hyper-parameters

epochs = 1
train_batch_size = 128


def train(model, train_loader, device, optimizer):
    with mlflow.start_run(run_name="Basic run"):
        model.train()
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()
        mlflow.log_param("Epochs", epochs)
        mlflow.log_param("Training batch size", train_batch_size)
        mlflow.log_metric("Loss", loss.item())

    #         if batch % 100 == 0:
    #             print(f"Train Epoch: {epoch} [{batch * len(data)}/{len(train_loader.dataset)}({(100. * batch / len(train_loader)):.0f}%)] \tLoss: {loss.item():.6f}")
    with open("results.json", 'w') as outfile:
        json.dump({"Batch size": train_batch_size, "Loss": loss.item()}, outfile)

    # print(f"\n Train Epoch: {epoch} \tLoss: {loss.item():.6f}")



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


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(model, train_loader, device, optimizer)
    # test(model, test_loader, device)
