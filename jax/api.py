import jax
from jax import random
from jax.tree_util import *
from typing import Callable, List, Any, Sequence, overload, Optional, Union
import jax.numpy as jnp
import gc


def jmp(f, W, M):
    "vmapped function of jvp for Jacobian-matrix product"
    _jvp = lambda s: jax.jvp(f, (W,), (s,))
    return jax.vmap(_jvp)(M)


def ggn(tangents, h):
    Jgh = (tangents @ h)[:,None]
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T

def ggn_off(t1, t2, h):
    return (t1 * h) @ t2.T - ((t1 @ h)[:,None]) @ ((t2 @ h)[None,:])

def random_split_like_tree(rng_key, target=None, treedef=None):
    "split key for a key for every leaf"
    if treedef is None:
        treedef = tree_structure(target)
    keys = random.split(rng_key, treedef.num_leaves)
    return tree_unflatten(treedef, keys)

def sample_v(rng, params, tangent_size):
    keys_tree = random_split_like_tree(rng, params)
    v = tree_map(
        lambda x,k: random.normal(k, (tangent_size,) + x.shape, x.dtype),
        params, keys_tree
    )
    #normalize, tangent-wise
    l2 = jnp.sqrt(sum(tree_leaves(
            jax.vmap(lambda v: tree_map(lambda x: jnp.sum(jnp.square(x)),v))(v)
        )))
    v = tree_map(lambda x: jax.vmap(lambda a,b:a/b)(x,l2), v)
    return v


def value_and_sofo_grad(
        fun: Callable,
        loss: Callable,
        argnums: int = 0,
        tangent_size: int =100,
        damping: float = 1E-5,
        classification: Optional[bool] = False,
    ) -> Callable[..., tuple[Any, Any]]:

    """ Create a function that evaluate "fun" and fish-forward gradient of "fun"
        
    Args:
        fun: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars. ``fun`` s answer should be concatenation
        of function on a batch of samples with mean function over the same batch.
        argnums: Optional, integer or sequence of integers. (default 0)
        tangent_size: Optional, number of random tangents to pass through jvp for 
        estimating fish-forward gradients. (default 100)

    Returns:
        A function with arguments same as ``fun`` plus a random key for generating
        the tangents, and returns a pair of values and gradients of ``fun``. 
    """

    def value_and_fish_grad_f(rng, params):
        rng, key = random.split(rng)
        v = sample_v(key, params, tangent_size)  

        outs, tangents_out = jmp(fun, params, v)    #tangents_out shape: t_size, b_size, out_size
        losses, vg = jmp(loss, outs[0], tangents_out)
        
        vggv = jax.lax.select(
                classification,
                jnp.mean(
                        jax.vmap(ggn, in_axes=(1,0))(tangents_out, jax.nn.softmax(outs[0], axis=-1))
                    , axis=0),
                jnp.mean(
                        jax.vmap(lambda t: t@t.T, in_axes=1)(tangents_out)
                    , axis=0))

        u,s,_ = jnp.linalg.svd(vggv)
        damped_s = jnp.sqrt(s) + damping # * jnp.max(jnp.sqrt(s))

        vggv_vg = (u / damped_s) @ (u.T @ vg)
        h = tree_map(lambda vs: jnp.dot(jnp.moveaxis(vs,0,-1), vggv_vg), v)
        return losses[0], h, jnp.max(s)

    return value_and_fish_grad_f

def jmp_apply(f, W, M):
        "vmapped function of jvp for Jacobian-matrix product"
        M_params, M_latents = M
        _jvp = lambda s,z: jax.jvp(f, W, (s, z), has_aux=True)
        return jax.vmap(_jvp)(M_params, M_latents)


def value_and_sofo_grad_temporal(
        rnn: Callable,
        loss: Callable,
        argnums: int = 0,
        tangent_size: int =100,
        damping: float = 1E-5,
        classification: Optional[bool] = False,
    ) -> Callable[..., tuple[Any, Any]]:

    """ Create a function that evaluate "fun" and fish-forward gradient of "fun" where the 
        function to be differentiated is a recurrent structure
        
    Args:
        rnn: Function to be differentiated. Its arguments at positions specified by
            ``argnums`` should be arrays, scalars. ``rnn`` is the recurrent model, that
            receives parameters, inputs, and outputs a tuple (carry, out) at each step.
        loss: Function to calculate the loss at each step, receives the predictions,
            which is usually the second output of ``rnn``, and also the labels.
        argnums: Optional, integer or sequence of integers. (default 0)
        tangent_size: Optional, number of random tangents to pass through jvp for 
        estimating fish-forward gradients. (default 100)

    Returns:
        A function with arguments same as ``fun`` plus a random key for generating
        the tangents, and returns a pair of values and gradients of ``fun``. 
    """
    #TODO: how to initilize z?

    def value_and_fish_grad_f_batch(z_init, batch):
        # batch:    will contain (batch_inputs, batch_labels),
        #           they should be of shape (time, batch, dim)
        # z_init:   should be of shape (batch, hidden_dim)

        def value_and_fish_grad_f(rng, params):
            rngs = random.split(rng, 2)
            v = sample_v(rngs[1], params, tangent_size)  
        
            def fun(carry, xs):
                latent, latent_tangents, losses, vg, vggv = carry
                inputs, labels = xs
            
                fun = lambda params, latent: rnn(params, latent, inputs)
                fun_loss = lambda logits: loss(logits, labels)

                latent_new, latent_tangents_out, logits  = jmp_apply(fun, (params, latent), (v, latent_tangents))
                losses_new, vg_new = jmp(fun_loss, latent_new[0], latent_tangents_out)
                
                #TODO: classification?
                #ybar = jax.nn.softmax(outs[0], axis=-1)
                #vggv = jnp.mean(
                #            jax.vmap(ggn, in_axes=(1,0))(tangents_out, ybar)
                #       , axis=0)

                vggv_new = jnp.mean(
                    jax.vmap(lambda t: t@t.T, in_axes=1)(latent_tangents_out),
                   axis=0)

                losses += losses_new[0]
                vg += vg_new
                vggv += vggv_new
                return (latent_new[0], latent_tangents_out, losses, vg, vggv), logits[0]
        
            (_, _, losses, vg, vggv), preds = jax.lax.scan(
                fun, init = (
                        z_init, 
                        jnp.zeros((tangent_size, *z_init.shape)), 
                        0.,
                        jnp.zeros((tangent_size,)),
                        jnp.zeros((tangent_size,tangent_size)),
                     ), xs = batch)

            u,s,_ = jnp.linalg.svd(vggv)
            damped_s = s + damping * jnp.max(s)

            vggv_vg = (u / damped_s) @ (u.T @ vg)
            h = tree_map(lambda vs: jnp.dot(jnp.moveaxis(vs,0,-1), vggv_vg), v)
            return losses, h, preds
        return value_and_fish_grad_f
    return value_and_fish_grad_f_batch
        
