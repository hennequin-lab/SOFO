import jax
from jax import random
from jax.tree_util import *
from typing import Callable, Any, Optional
import jax.numpy as jnp



def jmp(f, W, M, has_aux=False):
    """Batched Jacobian-vector products.

    Args:
        f (Callable): Function on which the primal is evaluated.
        W (jnp.ndarray): Primal.
        M (jnp.ndarray): Tangent.
        has_aux (bool, optional): Whether to return the output of function as first element. Defaults to False.

    Returns:
        jnp.ndarray or Tuple[jnp.ndarray, Any]:
            If has_aux is False:
                A tensor of shape (batch_size, *f(W).shape), the JVP results.
            If has_aux is True:
                A tuple (output, jvp_output), where output is f(W) and jvp_output is the batched JVP result.
    """    
    _jvp = lambda s: jax.jvp(f, (W,), (s,), has_aux=has_aux)
    return jax.vmap(_jvp)(M)

def jmp_pair(f, W, M, has_aux=False):
    """Batched Jacobian-vector products for a pair of primals.

    This computes the JVP of a function `f` with respect to two inputs (e.g., parameters and latents),
    batched over a set of tangent vectors.
    Args:
        f (Callable): Function on which the primal is evaluated.
        W (jnp.ndarray, jnp.ndarray): (primal, primal) pair.
        M (jnp.ndarray, jnp.ndarray): (tangent, tangent) pair.
        has_aux (bool, optional): Whether to return the output of function as first element. Defaults to False.

    Returns:
    Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        If `has_aux` is False:
            jnp.ndarray: Batched Jacobian-vector product.
        If `has_aux` is True:
            Tuple[jnp.ndarray, Any]: A tuple containing (output, auxiliary data).
    """
    M_1, M_2 = M
    _jvp = lambda M_1, M_2: jax.jvp(f, W, (M_1, M_2), has_aux=has_aux)
    return jax.vmap(_jvp)(M_1, M_2)


# GGN for cross entropy loss
def ggn_ce(tangents, h):
    """Generalised Gauss-Newton (GGN) matrixs for cross-entropy loss.

    Args:
        tangents (jnp.ndarray): Tangents associated with network output. size (k, batch_size, dim).
        h (jnp.ndarray): Predictions, usually probabilities of classes. size (dim,).

    Returns:
        jnp.ndarray: GGN matrix. size (k, k).
    """
    Jgh = (tangents @ h)[:,None]
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T  # (k, k)

# GGN for mean squared loss
def ggn_mse(tangents):
    """Generalised Gauss-Newton (GGN) matrixs for mean-squared loss.

    Args:
        tangents (jnp.ndarray): Tangents associated with network output. size (k, batch_size, dim).

    Returns:
        jnp.ndarray: GGN matrix. size (k, k).
    """
    return (tangents @ tangents.T)


def random_split_like_tree(rng_key, target=None, treedef=None):
    """Split key for a key for every leaf.

    Args:
        rng_key (jax.Array): A JAX PRNG key.
        target (PyTree, optional): A pytree to infer the tree structure from. 
                                   Required if `treedef` is not provided.
        treedef (TreeDef, optional): An explicit tree structure. If provided, `target` is ignored.

    Returns:
        PyTree: A pytree of PRNG keys with the same structure as `target` or `treedef`.
    """
    if treedef is None:
        treedef = tree_structure(target)
    keys = random.split(rng_key, treedef.num_leaves)
    return tree_unflatten(treedef, keys)

def sample_v(tangent_size, params, rng):
    """
    Samples a batch of random, normalized tangent vectors matching the structure of `params`.

    Each tangent vector is drawn from a standard normal distribution and normalized across
    the entire pytree (global L2 norm). The output is a pytree where each leaf has shape
    `(tangent_size, *x.shape)`.

    Args:
        tangent_size (int): The number of tangents/subspace dimension.
        params (PyTree): A pytree of parameters whose structure and shapes are used to sample tangents.
        rng (jax.Array): A JAX PRNG key.


    Returns:
        PyTree: A pytree with the same structure as `params`, where each leaf is a tensor of
                shape `(tangent_size, *leaf.shape)` representing a batch of normalized tangent vectors.
    """
    keys_tree = random_split_like_tree(rng, params)
    v = tree_map(
        lambda x,k: random.normal(k, (tangent_size,) + x.shape, x.dtype),
        params, keys_tree
    )
    # Normalize, tangent-wise
    l2 = jnp.sqrt(sum(tree_leaves(
            jax.vmap(lambda v: tree_map(lambda x: jnp.sum(jnp.square(x)),v))(v)
        )))
    v = tree_map(lambda x: jax.vmap(lambda a,b:a/b)(x,l2), v)
    return v


def value_and_sofo_grad(
        fun: Callable,
        loss: Callable,
        tangent_size: int =100,
        damping: float = 1E-5,
        classification: Optional[bool] = False,
    ) -> Callable[..., tuple[Any, Any]]:
    """SOFO forward pass to compute loss and gradient. 

    Args:
        fun (Callable): Forward pass of the network. ``fun`` s answer should be concatenation 
            of function on a batch of samples with mean function over the same batch.
        loss (Callable): Loss function.
        tangent_size (int, optional): Number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): Dampling parameter on ggn. Defaults to 1e-5.
        classification (bool, optional): Whether the task is classification. Defaults to False.
    """
    def value_and_fish_grad_f(rng, params):
        """Wrapper for the forward pass of the function.

        Args:
            rng (jax.random.PRNGKey): PRNG key used for sampling.
            params (PyTree): Model parameters — a pytree used as the input to `fun`.

        Returns:
            tuple:
                - loss_value (float): Scalar loss evaluated on the current batch.
                - h (PyTree): Gradient direction (same structure as `params`).
                - max_singular_value (float): Largest singular value of the approximated Fisher matrix,
                useful for monitoring curvature or condition number.
        """
        rng, key = random.split(rng)
        v = sample_v(tangent_size, params, key)  

        outs, tangents_out = jmp(fun, params, v)    #tangents_out shape: t_size, b_size, out_size
        losses, vg = jmp(loss, outs[0], tangents_out)
        
        vggv = jax.lax.select(
                classification,
                jnp.mean(
                        jax.vmap(ggn_ce, in_axes=(1,0))(tangents_out, jax.nn.softmax(outs[0], axis=-1))
                    , axis=0),
                jnp.mean(
                        jax.vmap(ggn_mse, in_axes=1)(tangents_out)
                    , axis=0))

        u,s,_ = jnp.linalg.svd(vggv)
        damped_s = s + damping * jnp.max(s)


        vggv_vg = (u / damped_s) @ (u.T @ vg)
        h = tree_map(lambda vs: jnp.dot(jnp.moveaxis(vs,0,-1), vggv_vg), v)
        return losses[0], h, jnp.max(s)

    return value_and_fish_grad_f



def value_and_sofo_grad_temporal(
        rnn: Callable,
        loss: Callable,
        tangent_size: int =100,
        damping: float = 1E-5,
        classification: Optional[bool] = False,
    ) -> Callable[..., tuple[Any, Any]]:
    """SOFO forward pass to compute loss and gradient. 

    Args:
        rnn (Callable): One-step update of the recurrent network. ``rnn`` s answer should be concatenation 
            of function on a batch of samples with mean function over the same batch.
        loss (Callable): Loss function.
        tangent_size (int, optional): Number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): Dampling parameter on ggn. Defaults to 1e-5.
        classification (bool, optional): Whether the task is classification. Defaults to False.
    """
    def value_and_grad_f_batch(z_init, batch):
        """Compute loss and gradient on a data batch.

        Args:
            z_init (jnp.ndarray): Initial state of the RNN.
            batch (tuple): A tuple (inputs, labels), where:
                - inputs (jnp.ndarray): Input sequence of shape (tmax, batch_size, input_dim).
                - labels (jnp.ndarray): Target sequence of shape (tmax, batch_size, output_dim) 
                  or (tmax, batch_size) for classification.
        """
        def wrapper(rng, params):
            """Wrapper for the forward pass of the RNN.

            Args:
                rng (jax.Array): A JAX PRNG key.
                params (PyTree): A pytree of parameters whose structure and shapes are used to sample tangents.

            Returns:
                tuple:
                    - loss_value (float): Scalar loss evaluated on the current batch.
                    - h (PyTree): Gradient direction (same structure as `params`).
                    - max_singular_value (float): Largest singular value of the GGN matrix,
                    useful for monitoring curvature or condition number.
            """
            rngs = random.split(rng, 2)
            v = sample_v(tangent_size, params, rngs[1])          
            def fun(carry, xs):
                """Recurrent function of the RNN.

                Args:
                    carry (tuple): (latent, latent_tangents, losses, vg, vggv) accumulated from previous step.
                    xs (tuple): (inputs, labels) at current step.

                Returns:
                    tuple:
                        - carry: (latent, latent_tangents, losses, vg, vggv) at current step.
                        - preds: The network output at the current iteration.
                """
                latent, latent_tangents, losses, vg, vggv = carry
                inputs, labels = xs
            
                fun = lambda params, latent: rnn(params, latent, inputs)
                fun_loss = lambda logits: loss(logits, labels)

                latent_new, latent_tangents_out, outs  = jmp_pair(fun, (params, latent), (v, latent_tangents), has_aux=True)
                [latent_primal, primal_out] = latent_new
                [new_latent_tangents_out, tangents_out] = latent_tangents_out
                losses_new, vg_new = jmp(fun_loss, primal_out[0], tangents_out)

                vggv_new = jax.lax.select(
                classification,
                jnp.mean(
                        jax.vmap(ggn_ce, in_axes=(1,0))(tangents_out, jax.nn.softmax(outs[0], axis=-1))
                    , axis=0),
                jnp.mean(
                        jax.vmap(ggn_mse, in_axes=1)(tangents_out)
                    , axis=0))


                losses += losses_new[0]
                vg += vg_new
                vggv += vggv_new
                return (latent_primal[0], new_latent_tangents_out, losses, vg, vggv), outs[0]
        
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
        return wrapper
    return value_and_grad_f_batch
        
