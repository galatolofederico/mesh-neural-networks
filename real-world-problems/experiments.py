import torch
import torchvision
import pickle
import argparse
import wandb
import numpy as np

def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_pkl = pickle.load(open(args.dataset_folder+"/train.pkl", "rb"))
    test_pkl = pickle.load(open(args.dataset_folder+"/test.pkl", "rb"))

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_pkl["x"]), torch.tensor(train_pkl["y"]))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_pkl["x"]), torch.tensor(test_pkl["y"]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    if args.arch == "mnn":
        A = torch.nn.Linear(args.inputs + args.mnn_hidden, args.mnn_hidden + args.outputs).to(device)
        optimizer = torch.optim.Adam(A.parameters(), lr=args.lr)
        if args.mnn_zero_prob > 0:
            A_mask = torch.rand(args.inputs + args.mnn_hidden, args.mnn_hidden + args.outputs)
            A_mask = A_mask <= args.mnn_zero_prob
            
            A.weight.data[A_mask.T] = 0

    elif args.arch == "mlp":
        net = torch.nn.Sequential(
            torch.nn.Linear(args.inputs, args.mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(args.mlp_hidden, args.mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(args.mlp_hidden, args.outputs)
        ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(0, args.epochs):
        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_y = batch_y.to(device)
            batch_X = batch_X.to(device)
            outs = None
            if args.arch == "mnn":
                state = torch.zeros(batch_X.shape[0], args.mnn_hidden).to(device)

                for t in range(0, args.mnn_ticks):
                    input = torch.cat((state[:, :args.mnn_hidden], batch_X), dim=1)
                    state = A(input)
                    state = torch.nn.functional.dropout(state, args.mnn_dropout_prob)

                outs = torch.softmax(state[:,args.mnn_hidden:], dim=1)

            elif args.arch == "mlp":
                outs = torch.softmax(net(batch_X), dim=1)

            loss = loss_fn(outs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            if args.mnn_zero_prob > 0:
                A.weight.grad[A_mask.T] = 0
            optimizer.step()
            
            acc = (outs.max(1).indices == batch_y).float().mean()
            
            if i == 0:
                print("epoch: %d    loss: %f    acc: %f" % (epoch, loss.item(), acc.item()))


    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    y_true = []
    y_pred = []

    for i, (batch_X, batch_y) in enumerate(test_loader):
        state = torch.zeros(batch_X.shape[0], args.mnn_hidden).to(device)
        batch_y = batch_y.to(device)
        batch_X = batch_X.to(device)
        outs = None

        if args.arch == "mnn":
            for t in range(0, args.mnn_ticks):
                input = torch.cat((state[:, :args.mnn_hidden], batch_X), dim=1)
                state = A(input)

            outs = torch.softmax(state[:,args.mnn_hidden:], dim=1)

        elif args.arch == "mlp":
            outs = torch.softmax(net(batch_X), dim=1)

        preds = outs.max(1).indices
        
        y_true.extend(batch_y.cpu().detach().numpy().tolist())
        y_pred.extend(preds.cpu().detach().numpy().tolist())

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return accuracy_score(y_true, y_pred)

def experiment(args):
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    print(args)
    results = []
    for i in range(0, args.experiments):
        result = train(args)
        print("run: %d result: %f" % (i, result))
        results.append(result)
    
    print(results)

    results = np.array(results)
    stats = {
        "results/best": results.max(),
        "results/mean": results.mean(),
        "results/var" : results.var()
        }
    
    print(stats)

    if args.wandb:
        wandb.log(stats)

    return stats

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, default="mnn")

    parser.add_argument("--mnn-hidden", type=int, default=20)
    parser.add_argument("--mnn-ticks", type=int, default=3)
    parser.add_argument("--mnn-dropout-prob", type=float, default=0.2)
    parser.add_argument("--mnn-zero-prob", type=float, default=0)

    parser.add_argument("--mlp-hidden", type=int, default=20)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--dataset-folder", type=str, default="datasets/mnist/")
    parser.add_argument("--inputs", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=10)
    
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mnn-mnist")
    parser.add_argument("--wandb-tag", type=str, default="")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    experiment(args)
