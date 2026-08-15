# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# from torch._six import inf
from math import inf
import logging
from termcolor import colored
import sys
import os
import time


@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                parameters = [p for p in parameters if p.grad is not None]

                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                for p in parameters:
                    if not torch.isfinite(p.grad).all():
                        # print(f"Gradient has nan/inf in {p._name}, skipping step")
                        optimizer.zero_grad()
                        self._scaler.update()
                        return None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

import torch
from math import inf

@torch.no_grad()
def ampscaler_get_grad_norm_v2(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """计算梯度范数，与 GradScaler 无关，支持 BF16/FP16。"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0., device=parameters[0].grad.device if parameters else 'cpu') # 确保返回Tensor在设备上
    
    device = parameters[0].grad.device
    
    if norm_type == inf:
        # 确保 max 操作是在相同 dtype 下进行的
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # 确保所有 norm 操作的结果 dtype 相同
        norm_values = [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        total_norm = torch.norm(torch.stack(norm_values), norm_type)
        
    return total_norm


class FlexibleScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, dtype=torch.float16):
        """
        初始化 Scaler。
        :param dtype: 当前混合精度训练使用的低精度类型，如 torch.float16 或 torch.bfloat16。
        """
        self.dtype = dtype
        # 只有在 FP16 模式下才需要 GradScaler
        self.use_scaler = self.dtype == torch.float16
        
        if self.use_scaler:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None # BF16 模式下不需要 GradScaler

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, retain_graph=False):
        
        if self.use_scaler:
            # --- FP16 模式：使用 GradScaler 进行缩放 ---
            self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
            
            if update_grad:
                assert parameters is not None
                parameters_with_grad = [p for p in parameters if p.grad is not None]

                # 必须先 unscale 才能检查 Inf/NaN 或进行梯度裁剪
                self._scaler.unscale_(optimizer) 

                # 检查 Inf/NaN (这是 GradScaler 内部逻辑的一部分，但在这里提前手动检查)
                for p in parameters_with_grad:
                    if not torch.isfinite(p.grad).all():
                        print(f"!!! BF16 ERROR: Inf/NaN gradient detected in parameter. dtype: {p.grad.dtype}")                        
                        optimizer.zero_grad()
                        self._scaler.update() # 保持状态不变，跳过本次更新
                        return None
                
                # 梯度裁剪和范数计算
                if clip_grad is not None:
                    norm = torch.nn.utils.clip_grad_norm_(parameters_with_grad, clip_grad)
                else:
                    norm = ampscaler_get_grad_norm_v2(parameters_with_grad)
                    
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
                
        else:
            # --- BF16 模式：不使用 GradScaler ---
            # 直接反向传播 (因为 BF16 不易下溢)
            loss.backward(create_graph=create_graph, retain_graph=retain_graph)
            
            if update_grad:
                assert parameters is not None
                parameters_with_grad = [p for p in parameters if p.grad is not None]

                # 检查 Inf/NaN (BF16 模式下通常不需要，但为了安全保留)
                for p in parameters_with_grad:
                    if not torch.isfinite(p.grad).all():
                        print(f"!!! BF16 ERROR: Inf/NaN gradient detected in parameter. dtype: {p.grad.dtype}")
                        optimizer.zero_grad()
                        # 注意：BF16 模式下没有 scaler.update()，直接返回 None
                        return None
                # 梯度裁剪和范数计算
                if clip_grad is not None:
                    norm = torch.nn.utils.clip_grad_norm_(parameters_with_grad, clip_grad)
                else:
                    norm = ampscaler_get_grad_norm_v2(parameters_with_grad)
                    
                optimizer.step()
            else:
                norm = None

        return norm

    def state_dict(self):
        if self.use_scaler:
            return self._scaler.state_dict()
        # BF16 模式下返回一个空字典或其他标识
        return {} 

    def load_state_dict(self, state_dict):
        if self.use_scaler and state_dict:
            self._scaler.load_state_dict(state_dict)
        # BF16 模式下忽略 load_state_dict

def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger