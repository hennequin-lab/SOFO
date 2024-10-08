from absl import app
from absl import logging
import os
## JAX
import jax
import jax.numpy as jnp
from jax import random
# Seeding for random operations
main_rng = random.PRNGKey(42)

import flax
from flax.training import train_state, checkpoints
import optax
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
from jax.tree_util import *

from api import value_and_sofo_grad
from mlp_mixer import MlpMixer

DATASET_PATH = "data/"
CHECKPOINT_PATH = "cifar10/checkpoint/"



def main(argv):
    
    opt = 'sgn'
    learning_rate = 0.05
    patch_size = (4,4)
    num_classes = 10
    num_blocks = 2
    hidden_dim = 128
    tokens_mlp_dim = 64
    channels_mlp_dim = 128
    tangent_size = 256
    damping = 1E-7
    weight_decay = 1E-6
    batch_size = 256
    num_epochs = 50
    momentum = 0.9
    eval_freq = 0.05


    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)


    # Transformations applied on each image => bring them into a numpy array
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    # We need to stack the batch elements
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)


    test_transform = image_to_numpy
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      image_to_numpy
                                     ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # We define a set of data loaders that we can use for training and validation
    train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=4,
                               persistent_workers=True)
    val_loader   = data.DataLoader(val_set,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=4,
                               persistent_workers=True)
    test_loader  = data.DataLoader(test_set,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=4,
                               persistent_workers=True)

    # initialize model
    model = MlpMixer(patch_size=patch_size,
                     num_classes=num_classes,
                     num_blocks=num_blocks,
                     hidden_dim=hidden_dim,
                     tokens_mlp_dim=tokens_mlp_dim,
                     channels_mlp_dim=channels_mlp_dim,)

    rng, init_rng, dropout_rng = random.split(main_rng, num=3)
    params = model.init({"params": init_rng, "dropout": dropout_rng}, jax.device_put(next(iter(train_loader))[0]), train=True)
    sizes = tree_leaves(tree_map(lambda x: x.reshape(-1).shape, params))
    indicies = tuple(jnp.cumsum(jnp.array(sizes))[:-1].tolist())
    num_params = np.sum(tree_leaves(tree_map(lambda x: np.prod(x.shape), params)))
    print("Num Params: ", num_params, "Tangents: ", tangent_size/num_params * 100, "%")
    

    if opt.lower() == 'adam':
        opt_class = optax.adam
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=learning_rate,
            boundaries_and_scales=
                {int(len(train_loader)*num_epochs*0.4): 0.5,
                 int(len(train_loader)*num_epochs*0.9): 0.5}
        )
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            opt_class(lr_schedule)
        )
    elif opt.lower() == 'sgn':
        opt_class = optax.sgd
        lr_schedule = optax.warmup_exponential_decay_schedule(
                                          init_value=0.0,
                                          peak_value=learning_rate, 
                                          warmup_steps=10,
                                          transition_steps=int(len(train_loader)*num_epochs)//50,
                                          decay_rate=0.9,
                                          end_value=0.03)
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            opt_class(lr_schedule, momentum=momentum)
        )
    
    opt_state = optimizer.init(params)

    def make_train_step(apply_fn, tx, tangent_size):
        @jax.jit
        def train_step(params, batch, opt_state, rng, damping):
            imgs, labels = batch
            rng, rng1, rng2 = jax.random.split(rng, num=3)
            
            output_fn = lambda params: apply_fn(params, imgs, rngs={'dropout': rng1}, train=True)[0] # output logits (B, 10)
            loss_fn = lambda logits: optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

            logits = apply_fn(params, imgs)[0]
            acc = (logits.argmax(axis=-1) == labels).mean()

            loss, grads, maxs = value_and_sofo_grad(
                    output_fn, loss_fn, tangent_size=tangent_size, damping=damping, classification=True,
                )(rng2, params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, acc, maxs
        return train_step

    def make_train_step_adam(apply_fn, tx):
        @jax.jit
        def train_step(params, batch, opt_state, rng):
            imgs, labels = batch
            def loss_fn(params): 
                logits = apply_fn(params, imgs, train=True)[0]
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                acc = (logits.argmax(axis=-1)==labels).mean()
                return loss, acc
            (loss, acc), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, acc
        return train_step
            
    
    def make_eval_step(apply_fn):
        @jax.jit
        def eval_step(params, batch):
            imgs, labels = batch
            logits = apply_fn(params, imgs, train=False)[0]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()

            return loss, acc
        return eval_step

    if opt=='sgn':
        train_step = make_train_step(model.apply, optimizer, tangent_size)
    else:
        train_step = make_train_step_adam(model.apply, optimizer)
    eval_step = make_eval_step(model.apply)
    training_log = []
    singular_values = []
    
    eval_loss, eval_acc = 2., 0.1
    
    itera = 0
    damp = jnp.array(1E-3)
    for epoch_idx in range(num_epochs):
        for i, batch in enumerate(train_loader):
            itera += 1
            rng, subkey = random.split(rng)
            

            params, opt_state, train_loss, train_acc, maxs = train_step(params, batch, opt_state, subkey, damp)

            if i % int(eval_freq*len(train_loader)) == 0:
                logging.info('Epoch{} [{}/{}]: maxs:{:.5f}, Train loss:{:.4f}, acc:{:.4f}, damp:{:.8f}, lr:{:.5f}'.format(
                    epoch_idx, i, len(train_loader), maxs.item(), train_loss.item(), train_acc.item(), damp.item(), lr_schedule(itera).item(),
                ))
                training_log.append([train_loss.item(), train_acc.item(), eval_loss, eval_acc, maxs.item()])
                damp = jnp.maximum(0.5 * damp, damping)

            #with open('logs/cifar10/sgn_1000_raw.txt', 'a') as f:
            #    f.write(','.join([str(round(a, 4)) for a in [train_loss.item(), train_acc.item(), eval_loss, eval_acc]]) + '\n')

        test_loss, test_acc = [],[]
        for batch in val_loader:
            loss, acc = eval_step(params, batch)
            test_loss.append(loss.item())
            test_acc.append(acc.item())
        eval_loss, eval_acc = np.mean(test_loss), np.mean(test_acc)
        logging.info('====Epoch{}: Eval loss:{:.4f}, acc:{:.4f}===='.format(epoch_idx, eval_loss, eval_acc))
    

if __name__ == '__main__':
    app.run(main)
