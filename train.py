import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import argparse
from utils.data_utils import get_loader
from model.model import ViT

np.random.seed(0)
torch.manual_seed(0)


def train(args, model, train_loader, device):
    """ Train the model """

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()
    for epoch in trange(args.n_epochs, desc="Training"):
        train_loss = 0.0
        for batch in train_loader:  # tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{args.n_epochs} loss: {train_loss:.2f}")

    torch.save(model.state_dict(), 'model.pt')


def test(model, test_loader, device):
    """ Test the model """

    criterion = CrossEntropyLoss()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--n_classes", default=10, type=int,
                        help="Number of classes in dataset.")
    parser.add_argument("--img_size", default=28, type=int,
                        help="Resolution size")
    parser.add_argument("--patch_size", default=4, type=int,
                        help="Patch size")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    # Training
    parser.add_argument("--n_epochs", default=10, type=int,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", default=0.005, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--h_dim", default=8, type=int,
                        help="Hidden dimension")
    parser.add_argument("--n_heads", default=2, type=int,
                        help="Number of self-attention heads")
    parser.add_argument("--n_blocks", default=2, type=int,
                        help="Number of encoder blocks")

    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # Data loading
    train_loader, test_loader, chw = get_loader(args)

    # Defining model
    model = ViT(n_classes=args.n_classes, patch_size=args.patch_size, hidden_dim=args.h_dim,
                n_heads=args.n_heads, n_blocks=args.n_blocks, chw=chw).to(device)

    # Training
    train(args, model, train_loader, device)

    # Test
    test(model, test_loader, device)


if __name__=="__main__":
    main()