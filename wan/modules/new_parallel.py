#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
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

import torch
import torch_npu
import torch.distributed as dist

MAX_TOKEN = 2147483647

def all_to_all_v(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    joint_tensor_key: torch.Tensor,
    joint_tensor_value: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    **kwargs
):
    scale = kwargs.get("scale", 1.0)
    algo = kwargs.get("algo", 0)
    self_attention = kwargs.get("self_attention", None)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # len_joint = joint_tensor_key.shape[1]
    global SEQ
    b, s, n, d = k.shape
    each_n = int(n // world_size)

    output_q_list = [torch.empty([b, s_i, each_n, d], device=q.device, dtype=q.dtype) for s_i in SEQ]
    output_k_list = [torch.empty([b, s_i, each_n, d], device=k.device, dtype=k.dtype) for s_i in SEQ]
    output_v_list = [torch.empty([b, s_i, each_n, d], device=v.device, dtype=v.dtype) for s_i in SEQ]
    q_list = [t.contiguous() for t in torch.tensor_split(q, world_size, scatter_dim)]
    k_list = [t.contiguous() for t in torch.tensor_split(k, world_size, scatter_dim)]
    v_list = [t.contiguous() for t in torch.tensor_split(v, world_size, scatter_dim)]    
    dist.all_to_all(output_q_list, q_list)
    dist.all_to_all(output_k_list, k_list)
    dist.all_to_all(output_v_list, v_list)

    query = torch.cat(output_q_list, dim=gather_dim).contiguous().transpose(1, 2)
    key = torch.cat(output_k_list, dim=gather_dim).contiguous().transpose(1, 2)
    value = torch.cat(output_v_list, dim=gather_dim).contiguous().transpose(1, 2)

    query_layer_list = query.split(1, dim=1)
    key_layer_list = key.split(1, dim=1)
    value_layer_list = value.split(1, dim=1)
    output = []
    for_loop = query.shape[1]
    for i in range(for_loop):
        if algo == 1:
            seqlen = torch.tensor([[query.shape[2]], [key.shape[2]]], dtype=torch.int32)
            intensors = [query_layer_list[i], key_layer_list[i], value_layer_list[i], seqlen]
            out = self_attention.forward(intensors)[0]
        elif algo == 0:
            out = torch_npu.npu_fusion_attention(
                query_layer_list[i],
                key_layer_list[i],
                value_layer_list[i],
                head_num=1,
                input_layout="BNSD",
                scale=query_layer_list[i].shape[-1]**-0.5,
                pre_tockens=MAX_TOKEN,
                next_tockens=MAX_TOKEN
            )[0]
        output.append(out)
    output = torch.cat(output, dim=1)
    output = output.transpose(1, 2)

    output_shape = [b, SEQ[0], each_n, d] if rank < world_size - 1 else [b, SEQ[-1], each_n, d]
    output_list = [torch.empty(output_shape, device=output.device, dtype=output.dtype) for _ in SEQ]

    SEQ_joint = [i for i in SEQ]
    output_con = [chunk.contiguous() for chunk in torch.split(output, SEQ_joint, dim=gather_dim)]

    dist.all_to_all(output_list, output_con)
    output = torch.cat(output_list, dim=scatter_dim)

    return output

SEQ = None

def split_sequence(input_, dim=1):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return input_
    
    tensor_list = torch.chunk(input_, world_size, dim=dim)
    global SEQ
    if not SEQ and input_.shape[dim] % world_size != 0:
        SEQ = [None] * world_size
        for i in range(world_size):
            SEQ[i] = tensor_list[i].shape[1]
    output = tensor_list[rank].contiguous()
    return output

def gather_sequence(input_, dim=1):
    input_ = input_.contiguous()
    world_size = dist.get_world_size()
    if world_size == 1:
         return input_
    
    global SEQ
    if not SEQ:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        b, s, d = input_.shape
        tensor_list = [torch.empty([b, s_i, d], device=input_.device, dtype=input_.dtype) for s_i in SEQ]
    dist.all_gather(tensor_list, input_)

    output = torch.cat(tensor_list, dim=dim)
    SEQ = None

    return output