from absl import app
from absl import flags
from absl import logging
import jax 
import flax.linen as nn
from jax.nn import log_softmax
import jax.numpy as jnp
from jax import random
from jax.tree_util import *
from flax.training import train_state
import optax
from typing import Callable, List, Any
import numpy.random as npr
import numpy as np

from datasets import mnist
from api import value_and_sofo_grad

def main(argv):
    logging.info("JAX procee: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    rng = random.PRNGKey(0)
    batch_size = 512
    output_size = 10
    input_size = 784
    iteration = 5000
    learning_rate = 0.8
    tangent_size = 512
    damping = 1E-7
    momentum = 0.9
    layers = [100,]

    ### load data
    train_images, train_labels, test_images, test_labels = mnist()
    train_size = train_images.shape[0]
    num_complete_batches, leftover = divmod(train_size, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(train_size)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()


    class MLP(nn.Module):
        hidden : List
        latent : int
        activation: Callable[..., Any] = nn.relu

        @nn.compact
        def __call__(self, x):
            for i, hidden in enumerate(self.hidden):
                x = nn.Dense(hidden, name='fc{}'.format(i))(x)
                x = self.activation(x)
            x = nn.Dense(self.latent, name='out')(x)
            return x

    ##### initialize model
    model = MLP(hidden=layers, latent=output_size)

    rng, subkey1, subkey2 = random.split(rng, 3)
    params = model.init(subkey1, random.normal(subkey2, (batch_size, input_size)))
    num_params = np.sum(tree_leaves(tree_map(lambda x: np.prod(x.shape), params)))
    print("Num Params: {}, Tangents: {} ({:.4f}%)".format(num_params, tangent_size, tangent_size/num_params*100))
    
    lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=iteration*2,
    )
    tx = optax.chain(
            optax.add_decayed_weights(1E-5),
            optax.sgd(learning_rate, momentum=momentum)
        )
    def accuracy(pred, targets):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(pred, axis=1)
        return jnp.mean(predicted_class == target_class)

    @jax.jit
    def train_step(params, data, rng, opt_state):
        batch_x, batch_y = data
        
        output_fn = lambda params: model.apply(params, batch_x)
        loss_fn = lambda logits: optax.softmax_cross_entropy(logits, batch_y).mean()

        loss, grads, _ = value_and_sofo_grad(
                output_fn, loss_fn, tangent_size=tangent_size, damping=damping, classification=True,
        )(rng, params)
        acc = accuracy(model.apply(params, batch_x), batch_y)
        
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, acc

    opt_state = tx.init(params)
    training_log = []
    singular_values = []
    for i in range(iteration):
        rng, subkey = random.split(rng)
        params, opt_state, train_loss, train_acc = train_step(params, next(batches), subkey, opt_state)
        epoch = i * batch_size / train_size

        if i % 25 == 0: 
            test_pred = model.apply(params, test_images)
            test_loss = optax.softmax_cross_entropy(test_pred,test_labels).mean()
            test_acc = accuracy(test_pred, test_labels)
            training_log.append([epoch,train_loss.item(),train_acc.item(),test_loss.item(),test_acc.item()])
        if i % 100 == 0:
            logging.info(
                    'Step {}: Train: loss:{:.4f}, acc:{:.3f}, \tTest: loss:{:.4f}, acc:{:.3f}'.format(i, train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item())
            )
    
    jnp.savez('logs/mnist/train_log_{}.npz'.format(tangent_size), train=jnp.asarray(training_log))
        

if __name__ == '__main__':

    app.run(main)


