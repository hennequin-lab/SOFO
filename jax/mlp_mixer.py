# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLP-Mixer model."""

from typing import Optional, Tuple
import einops
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp

class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
  """Mixer block layer."""
  tokens_mlp_dim: int
  channels_mlp_dim: int
  drop_p: float

  @nn.compact
  def __call__(self, x, *, train=False):
    #y = nn.LayerNorm()(x)
    y = jnp.swapaxes(x, 1, 2)
    y = MlpBlock(self.tokens_mlp_dim, name="token_mixing")(y)
    y = jnp.swapaxes(y, 1, 2)
    x = y * _stoch_depth_mask(x, self.drop_p, not train, self.make_rng)
    #y = nn.LayerNorm()(x)
    y = MlpBlock(self.channels_mlp_dim, name="channel_mixing")(x)
    return y * _stoch_depth_mask(x, self.drop_p, not train, self.make_rng)


class MlpMixer(nn.Module):
  """Mixer architecture."""
  patch_size: Tuple[int, int]
  num_classes: Optional[int]
  num_blocks: int
  hidden_dim: int
  tokens_mlp_dim: int
  channels_mlp_dim: int
  model_name: Optional[str] = None
  stoch_depth: float = 0.1

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}
    x = out["stem"] = nn.Conv(self.hidden_dim, self.patch_size,
                              strides=self.patch_size, name="stem")(image)
    x = out["input_tokens"] = einops.rearrange(x, "n h w c -> n (h w) c")
    for i in range(self.num_blocks):
      drop_p = (i / max(self.num_blocks - 1, 1)) * self.stoch_depth
      x = out[f"block_{i}"] = MixerBlock(
          self.tokens_mlp_dim, self.channels_mlp_dim, drop_p)(x, train=train)
    #x = nn.LayerNorm(name="pre_head_layer_norm")(x)
    x = out["pre_logits"] = jnp.mean(x, axis=1)
    if self.num_classes:
      x = out["logits"] = nn.Dense(
          self.num_classes, name="head")(x)
    return x, out


def _stoch_depth_mask(x, drop_p, deterministic, make_rng):
  if not deterministic and drop_p:
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return 1.0 - jax.random.bernoulli(make_rng("dropout"), drop_p, shape)
  return 1.0
