"""BTC（Bi-directional Transformer for Chords）のモデル定義。推論に要る部分だけ。

出典:
  - ChordMini https://github.com/ptnghia-j/ChordMini
    （commit aa6e3a8d7b017f082fd2aaff9329d5c26af49c03 の src/models/btc_model.py と
    src/models/common/transformer_modules.py、src/models/common/temporal_smoothing.py）
  - 元の実装 BTC-ISMIR19 https://github.com/jayg996/BTC-ISMIR19 （btc_model.py）

取り込むときに変えた点（重みの state_dict のキーと計算の中身は変えていない）:
  - マスクと位置の信号を register_buffer(persistent=False) にし、model.to(device) で一緒に動くようにした
  - 学習用の関数（predict、ファクトリなど）と attention map の返却を省いた

このファイルは torch を import するので、chordgen.chords の関数の中から遅延 import する。

ソースコードのライセンス（MIT。両リポジトリとも本文は同じ）:

Copyright (c) 2026 ChordMini contributors
Copyright (c) 2019 Jonggwon Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

重み（btc_model_best.pth、btc_model_large_voca.pt）はこの MIT の対象外。
CC-BY-NC-SA 4.0（非商用）として扱う（weights.py を参照）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ChordMini の config/ChordMini.yaml の model 節（BTC の既定値）と同じ値
BTC_CONFIG = {
    "feature_size": 144,
    "hidden_size": 128,
    "num_layers": 8,
    "num_heads": 4,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
    "seq_len": 108,
    "num_chords": 170,
}


def _gen_bias_mask(max_length: int) -> torch.Tensor:
    """未来側を -inf にするマスク [1, 1, L, L]。"""
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    return torch.from_numpy(np_mask).float().unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(max_length: int, channels: int, min_timescale=1.0, max_timescale=1.0e4) -> torch.Tensor:
    """正弦波の位置の信号 [1, L, C]。"""
    position = np.arange(max_length)
    num_timescales = channels // 2
    log_inc = np.log(float(max_timescale) / float(min_timescale)) / max(float(num_timescales) - 1, 1)
    inv = min_timescale * np.exp(np.arange(num_timescales).astype(np.float64) * -log_inc)
    scaled = np.expand_dims(position, 1) * np.expand_dims(inv, 0)
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant")
    return torch.from_numpy(signal.reshape([1, max_length, channels])).float()


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SoftmaxOutputLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)
        # 推論では使わないが、重みに含まれるので残す（strict に読み込むため）
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, batch_first=True, bidirectional=True)

    def forward(self, hidden):
        return self.output_projection(hidden)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, num_heads,
                 bias_mask=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        if bias_mask is not None:
            self.register_buffer("bias_mask", bias_mask, persistent=False)
        else:
            self.bias_mask = None
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, length, d = x.shape
        return x.view(b, length, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, length, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, length, h * d)

    def forward(self, queries, keys, values):
        queries = self._split_heads(self.query_linear(queries)) * self.query_scale
        keys = self._split_heads(self.key_linear(keys))
        values = self._split_heads(self.value_linear(values))
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if self.bias_mask is not None:
            logits = logits + self.bias_mask[:, :, : logits.shape[-2], : logits.shape[-1]].to(logits.dtype)
        weights = self.dropout(F.softmax(logits, dim=-1))
        return self.output_linear(self._merge_heads(torch.matmul(weights, values)))


class Conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super().__init__()
        padding = (kernel_size - 1, 0) if pad_type == "left" else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        return self.conv(self.pad(inputs.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_depth, filter_size, output_depth, layer_config="ll", padding="left", dropout=0.0):
        super().__init__()
        sizes = (
            [(input_depth, filter_size)]
            + [(filter_size, filter_size)] * (len(layer_config) - 2)
            + [(filter_size, output_depth)]
        )
        layers = []
        for lc, s in zip(list(layer_config), sizes):
            if lc == "l":
                layers.append(nn.Linear(*s))
            elif lc == "c":
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError(f"Unknown layer type {lc}")
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            # 元の実装どおり、最後の層のあとにも ReLU をかける
            x = self.dropout(self.relu(layer(x)))
        return x


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout
        )
        self.positionwise_convolution = PositionwiseFeedForward(
            hidden_size, filter_size, hidden_size, layer_config="cc", padding="both", dropout=relu_dropout
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, x):
        x_norm = self.layer_norm_mha(x)
        x = self.dropout(x + self.multi_head_attention(x_norm, x_norm, x_norm))
        y = self.positionwise_convolution(self.layer_norm_ffn(x))
        return self.dropout(x + y)


class _BiDirSelfAttention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()
        common = (hidden_size, total_key_depth, total_value_depth, filter_size, num_heads)
        drops = (layer_dropout, attention_dropout, relu_dropout)
        self.attn_block = _SelfAttentionBlock(*common, _gen_bias_mask(max_length), *drops)
        self.backward_attn_block = _SelfAttentionBlock(
            *common, torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3), *drops
        )
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))


class _BiDirSelfAttentionLayers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0,
                 relu_dropout=0.0):
        super().__init__()
        self.register_buffer("timing_signal", _gen_timing_signal(max_length, hidden_size), persistent=False)
        params = (hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                  layer_dropout, attention_dropout, relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[_BiDirSelfAttention(*params) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        x = self.embedding_proj(self.input_dropout(inputs))
        x = x + self.timing_signal[:, : x.shape[1], :].to(x.dtype)
        return self.layer_norm(self.self_attn_layers(x))


class BTCModel(nn.Module):
    """入力 [B, T, 144]（正規化した log-CQT）→ ロジット [B, T, 170]。T は seq_len 以下。"""

    def __init__(self, config: dict | None = None):
        super().__init__()
        c = dict(BTC_CONFIG, **(config or {}))
        self.timestep = int(c["seq_len"])
        self.self_attn_layers = _BiDirSelfAttentionLayers(
            c["feature_size"], c["hidden_size"], c["num_layers"], c["num_heads"], c["total_key_depth"],
            c["total_value_depth"], c["filter_size"], self.timestep, 0.2, 0.2, 0.2, 0.2,
        )
        self.output_layer = SoftmaxOutputLayer(hidden_size=c["hidden_size"], output_size=c["num_chords"])

    def forward(self, x):
        if x.shape[1] > self.timestep:
            raise ValueError(f"窓の長さ {x.shape[1]} が seq_len {self.timestep} を超えている")
        return self.output_layer(self.self_attn_layers(x))


def apply_temporal_smoothing(logits: torch.Tensor, kernel_size: int, use_gaussian: bool = False) -> torch.Tensor:
    """[B, T, C] のロジットを時間方向にならす（ChordMini の temporal_smoothing.py と同じ計算）。"""
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    x = logits.transpose(1, 2)  # [B, C, T]
    if use_gaussian:
        sigma = kernel_size / 6.0
        t = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1).expand(x.shape[1], 1, -1)
        y = F.conv1d(F.pad(x, (padding, padding), mode="replicate"), kernel, groups=x.shape[1])
    else:
        y = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=padding)
    return y.transpose(1, 2)
