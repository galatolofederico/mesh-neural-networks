import os
import argparse
import torch
import torchvision
from torchvision.utils import save_image
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--plots", action="store_true")

args = parser.parse_args()

assert not os.path.exists(args.output)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            torch.nn.Tanh()
        )
    
    def forward(self, x, features=False):
        f = self.encoder(x)
        y = self.decoder(f)
        return f if features else y

os.mkdir(args.output)
if args.plots:
    os.mkdir(args.output+"/plots")

train_loader = None
test_loader = None
model = None
loss_fn = None

if args.dataset == "mnist":
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("/tmp/mnist_data", train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("/tmp/mnist_data", train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=args.batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

elif args.dataset == "fashion":
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST("/tmp/fashion_mnist_data", train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                ])),
    batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST("/tmp/fashion_mnist_data", train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                ])),
    batch_size=args.batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()
else:
    raise Exception("Unknown dataset %s" % (args.dataset))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = AutoEncoder().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

for epoch in range(0, args.epochs):
    for it, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        
        batch_out = net(batch_x)
        loss = loss_fn(batch_out, batch_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print("epoch %d  loss: %.4f" % (epoch, loss.item()))


def dump_loader(loader, output):
    xs = []
    ys = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_features = net(batch_x, features=True).view(batch_x.shape[0], -1)

        for feature, y in zip(batch_features, batch_y):
            xs.append(feature.detach().cpu().numpy())
            ys.append(y.item())

    pickle.dump({
        "x": xs,
        "y": ys
    }, open(output, "wb"))

dump_loader(train_loader, args.output+"/train.pkl")
dump_loader(test_loader, args.output+"/test.pkl")

if args.plots:
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)    
        batch_out = net(batch_x)
        save_image(batch_out, args.output+"/plots/test_"+str(i)+".png")

