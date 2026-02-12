import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import numpy as np
import random
import os
from scipy.integrate import odeint

from api import value_and_sofo_grad_temporal

# --- Lorenz data ---
EPS = 1e-7


def generate_from_long(
    n_steps: int, n_trials: int, sigma=10.0, rho=28.0, beta=8.0 / 3.0
):

    tt = n_trials * n_steps * 100
    dt = 0.01
    duration = dt * (tt - 1)
    tspec = np.linspace(0, duration, num=tt)

    def lorenz(state, _):
        x, y, z = state[0], state[1], state[2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = (x * y) - beta * z
        return np.stack([dx, dy, dz], axis=-1)

    state0 = torch.randn(size=(3,))
    states = odeint(lorenz, state0, tspec)
    states = states.reshape(100 * n_trials, n_steps, 3)
    perm = torch.randperm(100 * n_trials)
    return states[perm, :]


def safe_norm(x, axis):
    norm = np.sum(np.square(x), axis=axis, keepdims=True)
    norm = np.where(condition=norm < EPS, x=1.0, y=norm)
    norm = np.sqrt(norm)
    return x / norm


# --- Model ---
class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_fn=F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden_dim
        self.activation_fn = activation_fn

        self.c = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, input_dim, bias=False)
        self.a = nn.Linear(input_dim, input_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h, inputs):
        # new_y = (h @ a) + (activation((h @c) + bias) @ w)
        linear_part = self.c(h) + self.b
        hidden_input = self.activation_fn(linear_part)
        hidden_part = self.w(hidden_input)
        new_h = self.a(h) + hidden_part
        return (new_h, new_h), new_h


def mse_k(pred, target, k):
    return ((pred[:, :k, :] - target[:, :k, :]) ** 2).sum(dim=(1, 2)).mean()


def r2_k(pred, target, k):
    baseline = target.mean(dim=1, keepdim=True)
    return 1 - mse_k(pred, target, k) / mse_k(baseline.expand_as(target), target, k)


def get_param_dict(model):
    return {
        name: param.clone().detach().requires_grad_(True)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(42)
    batch_size = 32
    input_dim = 3
    hidden_dim = 400
    sigma = 0.0125
    n_steps = 33
    n_trials = 500
    n_test = 100
    tangent_size = 128
    damping = 1e-5
    learning_rate = 0.01
    num_iterations = 3000
    k = 30
    eval_freq = 200

    ## Generate data
    bouts = generate_from_long(n_steps, n_trials + n_test)
    # normalize
    bouts = bouts.reshape(-1, 3)
    bouts = (bouts - np.mean(bouts, axis=0)) / np.sqrt(np.var(bouts, axis=0))
    bouts = bouts.reshape(-1, n_steps, 3)
    # add noise
    noise_bouts = bouts + np.random.randn(*bouts.shape) * sigma

    train_x, train_trajs = (
        bouts[: n_trials * 100, 0, :],
        noise_bouts[: n_trials * 100, 1:, :],
    )
    test_x, test_trajs = (
        bouts[n_trials * 100 :, 0, :],
        noise_bouts[n_trials * 100 :, 1:, :],
    )
    num_complete_batches, leftover = divmod(n_trials, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(n_trials)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_x[batch_idx], train_trajs[batch_idx]

    batches = data_stream()
    test_x, test_trajs = torch.from_numpy(test_x).float().to(device), torch.from_numpy(
        test_trajs
    ).float().to(device)

    model = RNNCell(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    init_params = get_param_dict(model)

    def rnn(params, latent, inputs):
        return functional_call(model, params, (latent, inputs))

    def predict(params, x):
        def fun(carry, _):
            h = carry
            (new_h, _), outs = rnn(params, h, None)
            return new_h, outs

        preds_list = []

        z = x
        for _ in range(n_steps - 1):
            z, preds = fun(z, None)
            preds_list.append(preds)

        preds_t = torch.stack(preds_list, dim=0)
        preds_final = torch.permute(preds_t, (1, 0, 2))
        return preds_final

    def train(batch_x, batch_y, params):
        batch_x, batch_y = torch.from_numpy(batch_x).float().to(
            device
        ), torch.from_numpy(batch_y).float().to(device)
        batch_label = torch.permute(batch_y, (1, 0, 2))

        loss_fn = lambda pred, label: torch.mean(
            torch.mean((pred - label) ** 2, dim=-1)
        )

        dummy_input = torch.zeros_like(batch_label)
        loss, grads, preds_list = value_and_sofo_grad_temporal(
            rnn,
            loss_fn,
            tangent_size=tangent_size,
            damping=damping,
            classification=False,
            device=device,
        )(z_init=batch_x, batch=(dummy_input, batch_label))(rng, params)

        with torch.no_grad():
            new_params = {k: params[k] - learning_rate * grads[k] for k in params}

        return new_params, loss, preds_list

    params = init_params
    training_log = []
    for i in range(num_iterations):
        model.train()
        x0, traj = next(batches)
        params, loss, _ = train(x0, traj, params)

        if i % 100 == 0:
            print(f"Iter {i},  Loss: {loss.item():.4f}")

        if i % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                test_pred = predict(params, test_x)
                test_loss = torch.mean(
                    torch.mean((test_pred - test_trajs) ** 2, dim=-1)
                )
                test_r2 = r2_k(test_pred, test_trajs, k).item()
                print(f"Iter {i}, Test Loss: {test_loss:.4f}, R²@{k}: {test_r2:.4f}")

            training_log.append([i, loss.item(), test_loss.item(), test_r2])
            np.savez(f"sofo_{tangent_size}.npz", np.asarray(training_log))


if __name__ == "__main__":
    main()
