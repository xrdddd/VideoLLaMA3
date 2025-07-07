# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import copy
import json
import os
import pathlib
import random
import re
import sys
import warnings
import traceback
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from pprint import pprint

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
from packaging import version
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

sys.path.append("./")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from videollama3.constants import (
    IGNORE_INDEX,
    NUM_FRAMES,
    DEFAULT_IMAGE_TOKEN,
    STREAM_MAX_FRAMES,
    STREAM_START_TOKEN,
    STREAM_END_TOKEN,
)
from videollama3.mm_utils import load_images, load_video, tokenizer_multimodal_token
from videollama3.model import *
from videollama3.videollama3_trainer import (
    VideoLLaMA3Trainer,
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from videollama3.model.processor import Videollama3Processor

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import sys
from transformers import AutoModelForCausalLM, AutoProcessor


# NOTE: transformers==4.46.3 is recommended for this script
model_path = "/teamspace/studios/this_studio/VideoLLaMA3/work_dirs/videollama3_qwen2.5_2b/stage_1"
model = Videollama3Qwen2ForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@torch.inference_mode()
def infer(conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# # Video conversation
# conversation = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "video": {"video_path": "./assets/cat_and_chicken.mp4", "fps": 1, "max_frames": 180}},
#             {"type": "text", "text": "What is the cat doing? Please describe the scene, the obejcts and the actions in detail."},
#         ]
#     },
# ]
# print(infer(conversation))

# Image conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": {"image_path": "./assets/sora.png"}},
            {"type": "text", "text": "Please describe the model?"},
        ]
    }
]
print(infer(conversation))

# # Mixed conversation
# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "video": {"video_path": "./assets/cat_and_chicken.mp4", "fps": 1, "max_frames": 180}},
#             {"type": "text", "text": "What is the relationship between the video and the following image?"},
#             {"type": "image", "image": {"image_path": "./assets/sora.png"}},
#         ]
#     }
# ]
# print(infer(conversation))

# Plain text conversation
conversation = [
    {
        "role": "user",
        "content": "What is the color of bananas?",
    }
]
print(infer(conversation))
