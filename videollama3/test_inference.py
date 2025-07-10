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
processor = Videollama3Processor.from_pretrained(model_path, trust_remote_code=True)
processor.image_processor = model.get_vision_encoder().image_processor #there should be better ways.

media_dir = '/teamspace/studios/this_studio/'

def _convert_normal(data_dict, data_folder):
        conversation = data_dict["conversations"]
        # data sanity check and repair
        start_idx = 0
        for sentence in conversation:
            if sentence["from"] == "human" or sentence["from"] == "system":
                break
            start_idx += 1
        if start_idx > 0:
            warnings.warn(f"Find {start_idx} non-user sentences at the beginning of the conversation, remove them automatically!")
            conversation = conversation[start_idx:]
        # assert len(conversation) > 1, f"Invalid conversation"

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            if all(not "<image>" in sentence["value"] for sentence in conversation):
                warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
                conversation[0]["value"] = "<image>" + conversation[0]["value"]
            image_file = data_dict['image']
            if isinstance(image_file, list):
                image_file = [os.path.join(data_folder, f) for f in image_file]
            else:
                image_file = os.path.join(data_folder, image_file)
            images = load_images(image_file)
        # elif 'video' in data_dict and data_dict['video'] is not None:
        #     modal = 'video'
        #     if all(not "<video>" in sentence["value"] for sentence in conversation):
        #         warnings.warn(f"Video tag not found in the conversation, add it automatically at the beginning!")
        #         conversation[0]["value"] = "<video>" + conversation[0]["value"]
        #     video_file = data_dict['video']
        #     if isinstance(video_file, list) and len(video_file) == 1:
        #         video_file = os.path.join(data_folder, video_file[0])
        #         images, timestamps = load_video(video_file, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
        #         images = [images]
        #     else:
        #         raise ValueError(f"Unsupported video format: {video_file}")
        else:
            modal = 'text'
            images = None

        messages = []
        for conv in conversation:
            if conv["from"] == "human":
                # replace video tag to image tag for unified processing
                # conv["value"] = conv["value"].replace("<video>", "<image>" * len(images))
                chunks = conv["value"].split("<image>" if modal == 'image' else "<video>")
                messages.append({
                    "role": "user",
                    "content": []
                })

                for chunk_idx in range(1, 2 * len(chunks)):
                    if chunk_idx % 2 == 1:
                        chunk = chunks[chunk_idx // 2].strip()
                        messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
                    else:
                        if modal == 'image':
                            messages[-1]["content"].append({"type": "image"})
                        # elif modal == 'video':
                        #     messages[-1]["content"].append({"type": "video", "num_frames": len(images[0]), "timestamps": timestamps})
            else:
                messages.append({
                    "role": "assistant",
                    "content": conv['value']
                })
        return modal, images, messages

def addtion_foo(vlprocessor, modal, data_dict, images):
    image_merge_size = 1
    video_merge_size = 2 

    if modal == "text":
        data_dict["pixel_values"] = None
        # unit_size = vlprocessor.image_processor.patch_size**2 * 3
        # data_dict["pixel_values"] = torch.zeros(
        #     image_merge_size**2, unit_size
        # )
        # data_dict["grid_sizes"] = torch.as_tensor(
        #     [
        #         [
        #             1,
        #             image_merge_size,
        #             image_merge_size,
        #         ]
        #     ]
        # )
        # data_dict["merge_sizes"] = torch.as_tensor(
        #     [image_merge_size]
        # )
    elif modal == "image" or modal == "video":
        assert (
            len(data_dict["pixel_values"]) > 0
            and len(data_dict["grid_sizes"]) > 0
        ), f"Invalid image data: {data_dict['images']}, {data_dict['grid_thws']}"

    data_dict["modals"] = [modal] * (len(images) if images is not None else 0)
    return data_dict

@torch.inference_mode()
def infer(processor, modal, images, messages):
    inputs = processor(
        images=images,
        text=messages,
        return_tensors="pt"
    )

    inputs = addtion_foo(processor, modal, inputs, images)

    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
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
# input = {
#   "conversations": [
#         {
#             "from": "human",
#             "value": "What is the color of bananas?"
#         }
#     ] 
# } 
# modal, images, messages = _convert_normal(input, media_dir)
# print(infer(processor, modal, images, messages))

# Image conversation
input = {
    "image": [
        "raw_res/LLaVA-OneVision-Data/image_1.png"
    ],
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nProvide a one-sentence caption for the provided image."
        }
    ] 
}
modal, images, messages = _convert_normal(input, media_dir)
print(infer(processor, modal, images, messages))