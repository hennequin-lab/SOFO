import torch
import torch.nn.functional as F


def jmp(f, W, M):
    "batched Jacobian vector products."
    _jvp = lambda s: torch.func.jvp(f, (W,), (s,))
    return torch.func.vmap(_jvp)(M)


def jmp_apply(f, W, M):
    "vmapped function of jvp for Jacobian-matrix product"
    M_params, M_latents = M
    _jvp = lambda s,z: torch.func.jvp(f, W, (s, z), has_aux=True)
    return torch.func.vmap(_jvp)(M_params, M_latents)


# GGN function
def ggn(tangents: torch.Tensor, h: torch.Tensor):
    # tangents: (k, batch_size, dim), h: (dim,)
    Jgh = (tangents @ h)[:,None]
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T  # (k, k)

# Sample normalized random tangents
def sample_v(tangent_size, params, device, rng):
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
    """
    fun: params tuple -> output tensor [batch, out_dim]
    loss: logits tensor -> scalar loss
    Returns a function rng, params -> (loss_val, sofo_grad, max_singular_val)
    """

    def wrapper(rng, params):
        v = sample_v(tangent_size, params, device, rng)
        
        # Compute model output and tangent outputs
        outs, tangents_out = jmp(fun, params,v)
        losses, vg = jmp(loss, outs[0], tangents_out)

        # Compute GGN matrix approx
        if classification:
            h_batch = F.softmax(outs[0], dim=-1)  # (batch_size, out_dim)
            ggn_vmapped = torch.func.vmap(ggn, in_dims=(1, 0))
            vggv = ggn_vmapped(tangents_out, h_batch).mean(dim=0)
        else:
            vggv = torch.mean(torch.func.vmap(lambda t: t @ t.T, in_dims=1)(tangents_out), dim=0)

        # SVD of GGN
        u, s, _ = torch.linalg.svd(vggv)
        damped_s = torch.sqrt(s) + damping

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
    """
    rnn: params tuple -> output tensor [batch, out_dim]
    loss: logits tensor -> scalar loss
    Returns a function rng, params -> (loss_val, sofo_grad, max_singular_val)
    """
    def value_and_grad_f_batch(z_init, batch):
        def wrapper(rng, params):
            v = sample_v(tangent_size, params, device, rng)

            def fun(carry, xs):
                latent, latent_tangents, losses, vg, vggv = carry
                inputs, labels = xs
            
                fun = lambda params, latent: rnn(params, latent, inputs)
                fun_loss = lambda preds: loss(preds, labels)

                # Compute next latent and tangents
                latent_new, latent_tangents_out, preds = jmp_apply(fun, (params, latent), (v, latent_tangents))

                [latent_primal, primal_out] = latent_new
                [new_latent_tangents_out, tangents_out] = latent_tangents_out
                losses_new, vg_new = jmp(fun_loss, primal_out[0], tangents_out)

                # Compute GGN matrix at this step
                if classification:
                    h_batch = F.softmax(primal_out[0], dim=-1)  # (batch_size, out_dim)
                    ggn_vmapped = torch.func.vmap(ggn, in_dims=(1, 0))
                    vggv_new = ggn_vmapped(tangents_out, h_batch).mean(dim=0)
                else:
                    vggv_new = torch.mean(torch.func.vmap(lambda t: t @ t.T, in_dims=1)(tangents_out), dim=0)
                
                losses += losses_new[0]
                vg += vg_new
                vggv += vggv_new
                return (latent_primal[0], new_latent_tangents_out, losses, vg, vggv), preds[0]
            
            # intialise quantities to be accumulated
            latent_tangent = torch.zeros((tangent_size,) + z_init.shape, device=device)
            losses = 0.
            vg = torch.zeros(tangent_size, device=device)
            vggv = torch.zeros((tangent_size, tangent_size), device=device)

            preds_list = []
            # need to convert batch_y to a list
            inputs, labels = batch 
            labels_lst = list(torch.unbind(labels, dim=0))
            inputs_lst = list(torch.unbind(inputs, dim=0))

            z = z_init
            for (inputs, labels) in zip(inputs_lst, labels_lst):
                (z, latent_tangent, losses, vg, vggv), preds = fun((z, latent_tangent, losses, vg, vggv), (inputs, labels))
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
