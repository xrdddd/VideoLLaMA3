# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, Qwen2Config,
                          Qwen2ForCausalLM, Qwen2Model)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from videollama3.constants import IGNORE_INDEX
from .videollama3_arch import Videollama3MetaForCausalLM, Videollama3MetaModel


class Videollama3Qwen2Config(Qwen2Config):
    model_type = "videollama3_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama3_qwen2"


class Videollama3Qwen2Model(Videollama3MetaModel, Qwen2Model):
    config_class = Videollama3Qwen2Config

    def __init__(self, config: Videollama3Qwen2Config):
        super(Videollama3Qwen2Model, self).__init__(config)


class Videollama3Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama3MetaForCausalLM):
    config_class = Videollama3Qwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Videollama3Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rank_steps_count = 0
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # NOTE: arguments are copied from transformers==4.46.3
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != IGNORE_INDEX
            shift_hidden_states = shift_hidden_states[mask]
            shift_labels = shift_labels[mask]

            if "num_items_in_batch" in loss_kwargs:
                reduction = "sum"
                num_items_in_batch = loss_kwargs["num_items_in_batch"]
            else:
                reduction = "mean"
                num_items_in_batch = None

            if getattr(self.config, "use_flash_loss", False):
                try:
                    from inf_cl.flash import FlashProb
                except ImportError:
                    raise ImportError(
                        "inf_cl is not installed. Please install it following https://github.com/DAMO-NLP-SG/Inf-CLIP."
                    )

                hidden_size = shift_hidden_states.size(-1)
                assert hidden_size % 256 == 0, f"hidden size ({hidden_size}) should be divisible by 256 when using flash loss"

                shift_hidden_states = shift_hidden_states.float()
                weight = self.lm_head.weight.float()

                lse = FlashProb.apply(
                    shift_hidden_states.reshape(shift_hidden_states.size(0), -1, 256),
                    weight.reshape(self.lm_head.weight.size(0), -1, 256),
                )
                numerator = torch.einsum("nd,nd->n", shift_hidden_states, weight[shift_labels])

                acc_op = torch.sum if reduction == "sum" else torch.mean
                loss = acc_op(-numerator + lse)

            else:
                shift_logits = self.lm_head(shift_hidden_states)
                loss = torch.nn.functional.cross_entropy(
                    shift_logits,
                    shift_labels,
                    reduction=reduction,
                )

            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch

        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        self.rank_steps_count+=1
        print('rank_steps_count:', self.rank_steps_count)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videollama3_qwen2", Videollama3Qwen2Config)
AutoModelForCausalLM.register(Videollama3Qwen2Config, Videollama3Qwen2ForCausalLM)
