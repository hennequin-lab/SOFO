import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.func import functional_call
import numpy as np

from api import value_and_sofo_grad


# --- Model ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def accuracy(pred, targets):
    predicted_class = pred.argmax(dim=1)
    return (predicted_class == targets).float().mean().item()


def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes=num_classes).float()


def get_param_dict(model):
    return {
        name: param.clone().detach().requires_grad_(True)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


# --- Training Loop ---


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = torch.Generator(device=device).manual_seed(42)

    eval_freq = 10
    n_epochs = 200
    batch_size = 256
    input_size = 784
    output_size = 10
    learning_rate = 0.01
    tangent_size = 340
    damping = 1e-3
    hidden_layers = [128]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLP(input_size, hidden_layers, output_size).to(device)
    init_params = get_param_dict(model)

    def train(batch_x, batch_y, params):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        output_fn = lambda params: functional_call(model, params, batch_x)
        loss_fn = lambda logits: F.cross_entropy(logits, batch_y).mean()

        loss, grads, _ = value_and_sofo_grad(
            output_fn,
            loss_fn,
            tangent_size=tangent_size,
            damping=damping,
            classification=True,
            device=device,
        )(rng, params)
        acc = accuracy(output_fn(params), batch_y)

        with torch.no_grad():
            new_params = {k: params[k] - learning_rate * grads[k] for k in params}

        return new_params, loss, acc

    params = init_params
    training_log = []
    for epoch in range(n_epochs):
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            params, loss, acc = train(batch_x, batch_y, params)
            if i % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}, Acc: {acc:.4f}"
                )

            # Evaluation
            if i % eval_freq == 0:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        logits = functional_call(model, params, (x,))
                        pred = logits.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                test_acc = correct / total
                epoch_ = epoch + i * batch_size / len(train_loader.dataset)

                training_log.append([epoch_, loss.item(), acc, test_acc])
                np.savez(
                    f"logs/mnist/sofo_{tangent_size}.npz", np.asarray(training_log)
                )
                print(f"Test accuracy after epoch {epoch_:.4f}: {test_acc:.4f}")


if __name__ == "__main__":
    main()
