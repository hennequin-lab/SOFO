from absl import app
from absl import flags
from absl import logging
import os
import jax 
import flax
import flax.linen as nn
from jax.nn import log_softmax, sigmoid, tanh
import jax.numpy as jnp
from jax.tree_util import *
from jax import random
from jax.tree_util import *
from flax.training import train_state
from flax.linen import initializers
import optax
from typing import Callable, List, Any, Sequence
import numpy.random as npr
from jax.experimental.ode import odeint
from functools import partial

from api import value_and_sofo_grad_temporal

EPS = 1E-7

def generate_from_long(
        rng, n_steps:int, n_trials:int, sigma=10., rho=28., beta=8./3.
    ):
    
    tt = n_trials * n_steps * 100
    dt = 0.01
    duration = dt * (tt-1)
    tspec = jnp.linspace(0, duration, num=tt)
    rng, key1, key2 = random.split(rng, 3)

    def lorenz(state, _):
        x,y,z = state[:,0], state[:,1], state[:,2]
        dx = sigma * (y-x)
        dy = x * (rho-z) - y
        dz = (x * y) - beta * z
        return jnp.stack([dx,dy,dz], axis=-1)

    state0 = random.normal(key1, shape=(1,3))
    states = odeint(lorenz, state0, tspec)
    states = states.reshape(100 * n_trials, n_steps, 3)
    perm = random.permutation(key2, 100 * n_trials)
    return states[perm, :]

def safe_norm(x, axis):
    norm = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    norm = jnp.where(condition=norm < EPS, x=1.0, y=norm)
    norm = jnp.sqrt(norm)
    return x/norm

class GRUCell(nn.recurrent.RNNCellBase):
    dim: int
    hidden: int
    activation_fn: Callable[..., Any]= nn.relu

    @nn.compact
    def __call__(self, carry, _):
        h = carry
        bias = self.param(
                'bias', initializers.zeros_init(),
                (self.dim,)
            )
        Wl = self.param('linear', initializers.variance_scaling(
                                scale=0.2, mode='fan_in', distribution='truncated_normal'),
                        (self.dim, self.dim))
        Wh = self.param('hidden', initializers.variance_scaling(
                                scale=0.2, mode='fan_in', distribution='truncated_normal'),
                        (self.hidden, self.dim))
        C = self.param('expand', initializers.variance_scaling(
                                scale=0.2, mode='fan_in', distribution='truncated_normal'),
                        (self.dim, self.hidden))
 
        new_h = h@Wl + self.activation_fn(h@C)@Wh + bias
        Wout = self.param('readout', initializers.variance_scaling(
                                scale=0.2, mode='fan_in', distribution='truncated_normal'),
                            (self.dim, self.dim))
        outs = self.activation_fn(new_h) @ Wout
        return (new_h, outs), outs


def make_train_step(apply_fn, tx, sigma, tangent_size, damping):
    value_and_grad = value_and_sofo_grad_temporal(
            apply_fn, 
            lambda pred, label: jnp.mean(jnp.square(pred-label).mean(axis=-1)),
            tangent_size=tangent_size,
            damping=damping
        )

    @jax.jit
    def train_step(params, batch, rng, opt_state):
        batch_x, batch_y = batch
        batch_label = batch_y.swapaxes(0,1)

        rng, sample_key = random.split(rng)

        dummy_input = jnp.zeros_like(batch_label)
        loss_value, grads, pred_y = value_and_grad(
                z_init = batch_x, 
                batch = (dummy_input, batch_label)
            )(sample_key, params)

        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, pred_y.swapaxes(0,1), batch_y
    return train_step

def main(argv):
    logging.info("JAX procee: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    rng = random.PRNGKey(0)
    batch_size = 256
    latent_dim = 400
    iteration = 8000
    learning_rate = 0.05
    tangent_size = 128
    damping = 1E-5
    sigma = 0.0125
    n_steps = 51
    n_trials = 400
    n_test = 100
    eval_freq = 20
    k = 30
    
    ### load data
    if os.path.exists("trajs.npy"):
        bouts = jnp.load("trajs.npy")
    else:
        rng, subkey = random.split(rng)
        bouts = generate_from_long(subkey, n_steps, n_trials + n_test)
        jnp.save("trajs.npy", bouts)
    # normalize
    rng, subkey = random.split(rng)
    bouts = bouts.reshape(-1,3)
    bouts = (bouts - jnp.mean(bouts, axis=0)) / jnp.sqrt(jnp.var(bouts, axis=0))
    bouts = bouts.reshape(-1, n_steps, 3)
    
    # add noise
    rng, subkey = random.split(rng)
    noise_bouts = bouts + random.normal(subkey,bouts.shape,bouts.dtype) * sigma

    train_x, train_trajs = bouts[:n_trials*100, 0, :], noise_bouts[:n_trials*100, 1:, :]
    test_x, test_trajs = bouts[n_trials*100:, 0,:], noise_bouts[n_trials*100:, 1:,:]
    num_complete_batches, leftover = divmod(n_trials, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(n_trials)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i+1) * batch_size]
                yield train_x[batch_idx], train_trajs[batch_idx]

    batches = data_stream()
    def msek(pred, obs, k): 
        return jnp.square(pred[:,:k,:]-obs[:,:k,:]).sum(axis=(-1,-2)).mean()
    def r2k(pred, obs, k):
        mean = obs.mean(axis=1, keepdims=True) # B,H,3
        return 1. - msek(pred, obs, k) / msek(mean, obs, k)
    
    ### intialize model
    model = GRUCell(dim=3,
                hidden=latent_dim,)

    rng, subkey1, subkey2 = random.split(rng, 3)
    y, params = model.init_with_output(subkey, random.normal(subkey2, (batch_size, 3)), None)
        
    def predict(params, x):
        def fun(carry,_):
            h = carry
            (new_h, _), outs = model.apply(params, h, None)
            return new_h, outs

        _, results = jax.lax.scan(
                fun, init=x, xs=None,
                length=n_steps-1)
        return results.swapaxes(0,1)
        

    for path, value in flax.traverse_util.flatten_dict(params).items():
        print(path, value.shape)
    print("total size: ", sum(tree_leaves(tree_map(lambda x: jnp.size(x), params))))
    
        
    schedule = optax.exponential_decay(
                                       init_value=learning_rate,
                                       transition_steps=500,
                                       decay_rate=0.95,
                                       end_value=0.00001)
    tx = optax.sgd(learning_rate = schedule)
    opt_state = tx.init(params)
    train_step = make_train_step(model.apply, tx, sigma, tangent_size, damping)
    training_log = []

    _, _, loss_value, pred_y, batch_y = train_step(params, next(batches), subkey, opt_state)
    for i in range(iteration):
        rng, subkey = random.split(rng)
        
        if i % eval_freq == 0:
            test_pred = predict(params, test_x)
            test_loss = jnp.mean(jnp.square(test_pred-test_trajs).mean(axis=-1).sum(axis=-1))
            test_msek = msek(test_pred, test_trajs, k)
            logging.info(
                        'Step {}: Train loss:{:.4f}, Test loss:{:.4f}, msek:{:.3f}, lr:{:.4f}, tangent:{:d}'.format(
                    i, loss_value.item(), test_loss.item(), test_msek.item(), schedule(i), tangent_size,
                ))

        params, opt_state, loss_value, pred_y, batch_y = train_step(params, next(batches), subkey, opt_state)
        epoch = i * batch_size / n_trials
        training_log.append([epoch, loss_value.item(), test_loss.item(), test_msek.item()])

    jnp.savez("logs/lorenz/fish_ts{}.npz".format(tangent_size), train=jnp.asarray(training_log))
    

if __name__ == '__main__':
    app.run(main)
