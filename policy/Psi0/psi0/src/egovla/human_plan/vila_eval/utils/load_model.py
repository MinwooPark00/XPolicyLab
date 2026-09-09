from transformers import HfArgumentParser, AutoTokenizer, AutoConfig, LlamaForCausalLM
from human_plan.vila_train.args import (
    VLATrainingArguments, VLAModelArguments, VLADataArguments
)
import numpy as np
from human_plan.utils.action_tokenizer import build_action_tokenizer

from llava.model.builder import load_pretrained_model

from llava.mm_utils import get_model_name_from_path

from llava import conversation as conversation_lib
from llava.train.train import (
    maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    get_mm_adapter_state_maybe_zero_3,
    find_all_linear_names,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize
)
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

def load_model_eval(model_args, data_args, training_args):
  training_args.run_name = training_args.output_dir.split("/")[-1]

  data_args.is_multimodal = True
  checkpoint_path = model_args.model_name_or_path
  print(checkpoint_path)
  model_base = "checkpoints/ego_vla_checkpoint"

  model_name = get_model_name_from_path(checkpoint_path)

  tokenizer, model, image_processor, context_len = load_pretrained_model(
      checkpoint_path, model_name, model_base,
      # load_4bit=False, load_8bit=False, use_flash_attn=True
  )
  if tokenizer.bos_token is None:
      smart_tokenizer_and_embedding_resize(
          special_tokens_dict=dict(bos_token="[BOS]"),
          tokenizer=tokenizer,
          model=model.llm,
      )

  # @yunhao: may move this block into method "build_llm"
  tokenizer.pad_token = tokenizer.unk_token
  if tokenizer.pad_token is None:
      smart_tokenizer_and_embedding_resize(
          special_tokens_dict=dict(pad_token="[PAD]"),
          tokenizer=tokenizer,
          model=model.llm,
      )
  if model_args.version in conversation_lib.conv_templates:
      conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
  else:
      conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


  model_args.predict_future_step = data_args.predict_future_step
  data_args.action_tokenizer = build_action_tokenizer(
      model_args.action_tokenizer, tokenizer, model_args
  )
  model.config.invalid_token_idx = data_args.action_tokenizer.invalid_token_idx
  model.config.input_placeholder_token_idx = data_args.action_tokenizer.input_placeholder_token_idx
  model.config.input_placeholder_start_token_idx = data_args.action_tokenizer.input_placeholder_start_token_idx
  model.config.input_placeholder_end_token_idx = data_args.action_tokenizer.input_placeholder_end_token_idx

  model.config.merge_hand = data_args.merge_hand
  data_args.sep_query_token = model_args.sep_query_token
  
  data_args.traj_action_output_dim = model.traj_decoder.out_dim
  model.config.invalid_token_weight = training_args.invalid_token_weight

  data_args.image_processor = image_processor

  vision_tower = model.get_vision_tower()
  if vision_tower is not None:
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    if hasattr(data_args, "num_video_frames") and data_args.num_video_frames != None:
        model.config.num_video_frames = data_args.num_video_frames
    else:
        model.config.num_video_frames = 8

    if hasattr(data_args, "fps"):
        model.config.fps = data_args.fps
    else:
        model.config.fps = 0.0

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    if model_args.mm_use_im_start_end:
        num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    assert not model_args.mm_use_im_patch_token

    model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
    model.config.time_token_format = data_args.time_token_format = model_args.time_token_format
    if model_args.num_time_tokens > 0:
        time_tokens = [model.config.time_token_format.format(t=t) for t in range(model.config.num_time_tokens)]
        num_new_tokens = tokenizer.add_tokens(time_tokens)
        assert len(time_tokens) == num_new_tokens or num_new_tokens == 0
        model.resize_token_embeddings(len(tokenizer))
        model.config.time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
    else:
        model.config.time_token_ids = []
    model.config.soft_ce_std = model_args.soft_ce_std
  
  # kentang-mit@: It will be useful in on-the-fly packing
  model.llm.pad_token_id = tokenizer.pad_token_id
  model.llm.config.tokenizer_padding_side = tokenizer.padding_side
  model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length


  return model, tokenizer, model_args, data_args, training_args
