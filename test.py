import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from model import Network

epochs = 1
test_batch_size = 1000


def test(model, test_loader, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)}({(100. * correct / len(test_loader.dataset)):.2f}%)\n")
    print('----------------------------------------------------')


test_set = torchvision.datasets.FashionMNIST('./data/FashionMNIST',
                                              train = False,
                                              download = True,
                                             transform = transforms.Compose([
                                                 transforms.ToTensor()
                                             ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)

for epoch in range(1, epochs + 1):
    test(model, test_loader, device)