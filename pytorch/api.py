import torch
import torch.nn.functional as F

# Jacobian-vector product
def jmp(f, W, M, has_aux=False):
    """Batched Jacobian-vector products

    Args:
        f (function): function on which the primal is evaluated.
        W (torch.Tensor): primal.
        M (torch.Tensor): tangent.
        has_aux (bool, optional): whether to return the output of function as first element. Defaults to False.
    Returns:
        torch.Tensor or Tuple[torch.Tensor, Any]:
            If has_aux is False:
                A tensor of shape (batch_size, *f(W).shape), the JVP results.
            If has_aux is True:
                A tuple (output, jvp_output), where output is f(W) and jvp_output is the batched JVP result.
    """
    _jvp = lambda s: torch.func.jvp(f, (W,), (s,), has_aux)
    return torch.func.vmap(_jvp)(M)


def jmp_pair(f, W, M, has_aux=False):
    """Batched Jacobian-vector products for a pair of primals.

    This computes the JVP of a function `f` with respect to two inputs (e.g., parameters and latents),
    batched over a set of tangent vectors.
    Args:
        f (function): function on which the primal is evaluated.
        W (torch.Tensor, torch.Tensor): (primal, primal) pair.
        M (torch.Tensor, torch.Tensor): (tangent, tangent) pair.
        has_aux (bool, optional): whether to return the output of function as first element. Defaults to False.

    Returns:
    Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        If `has_aux` is False:
            torch.Tensor: Batched Jacobian-vector product.
        If `has_aux` is True:
            Tuple[torch.Tensor, Any]: A tuple containing (output, auxiliary data).
    """
    M_1, M_2 = M
    _jvp = lambda M_1, M_2: torch.func.jvp(f, W, (M_1, M_2), has_aux)
    return torch.func.vmap(_jvp)(M_1, M_2)


# GGN for cross entropy loss
def ggn_ce(tangents, h):
    """Generalised Gauss-Newton (GGN) matrixs for cross-entropy loss.

    Args:
        tangents (torch.Tensor): tangents associated with network output. size (k, batch_size, dim).
        h (torch.Tensor): predictions, usually probabilities of classes. size (dim,).

    Returns:
        torch.Tensor: GGN matrix. size (k, k).
    """
    Jgh = (tangents @ h)[:,None]
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T  

# GGN for mean squared loss
def ggn_mse(tangents: torch.Tensor):
    """Generalised Gauss-Newton (GGN) matrixs for mean-squared loss.

    Args:
        tangents (torch.Tensor): tangents associated with network output. size (k, batch_size, dim).

    Returns:
        torch.Tensor: GGN matrix. size (k, k).
    """
    return torch.func.vmap(lambda t: t @ t.T, in_dims=1)(tangents)

def sample_v(tangent_size, params, device, rng):
    """Sample tangents associated with parameters.

    Args:
        tangent_size (int): number of tangents/subspace dimension.
        params (dict): parameters to train.
        device (torch.device): device on which to generate tensors.
        rng (torch.Generator): a pseudorandom number generator for sampling.

    Returns:
        dict: key is the parameter name and value is its tangent.
    """
    v = {}
    for name, p in params.items():
        shape = (tangent_size,) + p.shape
        sample = torch.randn(shape, device=device, dtype=p.dtype, generator=rng)
        v[name] = sample

    # Compute total squared norm across all parameters for each tangent
    squared_norms = torch.zeros(tangent_size, device=device)
    for vi in v.values():
        squared_norms += vi.flatten(1).pow(2).sum(dim=1)
    norm = torch.sqrt(squared_norms)  # shape: [tangent_size]

    # Normalize each tangent
    normalized_v = {
        name: vi / norm.view(-1, *([1] * (vi.ndim - 1)))
        for name, vi in v.items()
    }

    return normalized_v



def value_and_sofo_grad(fun, loss, tangent_size=100, damping=1e-5, classification=False, device="cpu"):
    """SOFO forward pass to compute loss and gradient. 

    Args:
        fun (function): forward pass of the network.
        loss (function): loss function.
        tangent_size (int, optional): number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): dampling parameter on ggn. Defaults to 1e-5.
        classification (bool, optional): whether the task is classification. Defaults to False.
        device (str, optional): device on which the network is run. Defaults to "cpu".
    """
    def wrapper(rng, params):
        """Wrapper for the forward pass of the function.

        Args:
            rng (int): pseudorandom number generator for sampling tangents.
            params (dict): network params where key is the name and value is the params (primal).

        Returns:
            tuple:
                - loss_value (float): Scalar loss evaluated on the current batch.
                - h (dict): Gradient direction (same structure as `params`).
                - max_singular_value (float): Largest singular value of the GGN matrix,
                useful for monitoring curvature or condition number.
        """
        # Sample tangents associated with params
        v = sample_v(tangent_size, params, device, rng)
        
        # Compute model output and tangent outputs
        outs, tangents_out = jmp(fun, params,v)
        losses, vg = jmp(loss, outs[0], tangents_out)

        # Compute GGN matrix approx
        if classification:
            h_batch = F.softmax(outs[0], dim=-1)  # (batch_size, out_dim)
            ggn_vmapped = torch.func.vmap(ggn_ce, in_dims=(1, 0))
            vggv = ggn_vmapped(tangents_out, h_batch).mean(dim=0)
        else:
            vggv = ggn_mse(tangents_out).mean(dim=0)

        # SVD of GGN
        u, s, _ = torch.linalg.svd(vggv)
        damped_s = s + damping * torch.max(s)

        # Compute damped inverse times vg
        vggv_vg = (u / damped_s) @ (u.T @ vg)

        h = {}
        for name, vs in v.items(): 
            # vs: tensor of shape [tangent_size, *param_shape]
            # vggv_vg: tensor of shape [tangent_size]
            # Move tangent dimension to the last
            permuted_vs = vs.movedim(0, -1)  # shape [..., tangent_size]
            # Contract with vggv_vg over last dimension
            h[name] = torch.tensordot(permuted_vs, vggv_vg, dims=([-1], [0]))  # shape: param_shape

        return losses[0].detach(), h, s.max().detach()
    return wrapper


def value_and_sofo_grad_temporal(rnn, loss, tangent_size=100, damping=1e-5, classification=False, device="cpu"):
    """SOFO forward pass on a recurrent neural network to compute loss and gradient. 

    Args:
        rnn (function): one-step update of the recurrent network.
        loss (function): loss function.
        tangent_size (int, optional): number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): dampling parameter on ggn. Defaults to 1e-5.
        classification (bool, optional): whether the task is classification. Defaults to False.
        device (str, optional): device on which the network is run. Defaults to "cpu".
    """
    def value_and_grad_f_batch(z_init, batch):
        """Compute loss and gradient on a data batch.

        Args:
            z_init (torch.Tensor): initial state of the RNN.
            batch (tuple): A tuple (inputs, labels), where:
                - inputs (torch.Tensor): Input sequence of shape (tmax, batch_size, input_dim).
                - labels (torch.Tensor): Target sequence of shape (tmax, batch_size, output_dim) 
                  or (tmax, batch_size) for classification.        """
        def wrapper(rng, params):
            """Wrapper for the forward pass of the RNN.

            Args:
                rng (int): pseudorandom number generator for sampling tangents.
                params (dict): network params where key is the name and value is the params (primal).

            Returns:
                tuple:
                    - loss_value (float): Scalar loss evaluated on the current batch.
                    - h (dict): Gradient direction (same structure as `params`).
                    - max_singular_value (float): Largest singular value of the GGN matrix,
                    useful for monitoring curvature or condition number.
            """
            v = sample_v(tangent_size, params, device, rng)

            def fun(carry, xs, tmax):
                """One-step recurrent function of the RNN.

                Args:
                    carry (tuple): (latent, latent_tangents, losses, vg, vggv) accumulated from previous step.
                    xs (tuple): (inputs, labels) at current step.
                    tmax (int): total number of steps.

                Returns:
                    tuple:
                        - carry: (latent, latent_tangents, losses, vg, vggv) at current step.
                        - preds: The network output at the current iteration.
                """
                latent, latent_tangents, losses, vg, vggv = carry
                inputs, labels = xs
            
                fun = lambda params, latent: rnn(params, latent, inputs)
                fun_loss = lambda preds: loss(preds, labels)

                # Compute next latent and tangents
                latent_new, latent_tangents_out, preds = jmp_pair(fun, (params, latent), (v, latent_tangents), has_aux=True)

                [latent_primal, primal_out] = latent_new
                [new_latent_tangents_out, tangents_out] = latent_tangents_out
                losses_new, vg_new = jmp(fun_loss, primal_out[0], tangents_out)

                # Compute GGN matrix at this step
                if classification:
                    h_batch = F.softmax(primal_out[0], dim=-1)  # (batch_size, out_dim)
                    ggn_vmapped = torch.func.vmap(ggn_ce, in_dims=(1, 0))
                    vggv_new = ggn_vmapped(tangents_out, h_batch).mean(dim=0)
                else:
                    vggv_new = ggn_mse(tangents_out).mean(dim=0)
                                    
                losses += losses_new[0]/tmax
                vg += vg_new/tmax
                vggv += vggv_new/tmax
                return (latent_primal[0], new_latent_tangents_out, losses, vg, vggv), preds[0]
            
            # Intialise quantities to be accumulated
            latent_tangent = torch.zeros((tangent_size,) + z_init.shape, device=device)
            losses = 0.
            vg = torch.zeros(tangent_size, device=device)
            vggv = torch.zeros((tangent_size, tangent_size), device=device)

            preds_list = []
            # Need to convert batch_y to a list
            inputs, labels = batch 
            labels_lst = list(torch.unbind(labels, dim=0))
            inputs_lst = list(torch.unbind(inputs, dim=0))
            tmax = len(inputs_lst)
            # Recurrent pass through the RNN
            z = z_init
            for (inputs, labels) in zip(inputs_lst, labels_lst):
                (z, latent_tangent, losses, vg, vggv), preds = fun((z, latent_tangent, losses, vg, vggv), (inputs, labels), tmax)
                preds_list.append(preds)

            preds_t = torch.stack(preds_list, dim=0)
            preds_final = torch.permute(preds_t, (1,0,2))

            # SVD of GGN
            u, s, _ = torch.linalg.svd(vggv)
            damped_s = s + damping * torch.max(s)

            # Compute damped inverse times vg
            vggv_vg = (u / damped_s) @ (u.T @ vg)

            h = {}
            for name, vs in v.items(): 
                # vs: tensor of shape [tangent_size, *param_shape]
                # vggv_vg: tensor of shape [tangent_size]
                # Move tangent dimension to the last
                permuted_vs = vs.movedim(0, -1)  # shape [..., tangent_size]
                # Contract with vggv_vg over last dimension
                h[name] = torch.tensordot(permuted_vs, vggv_vg, dims=([-1], [0]))  # shape: param_shape

            return losses.detach(), h, preds_final
        return wrapper
    return value_and_grad_f_batch
