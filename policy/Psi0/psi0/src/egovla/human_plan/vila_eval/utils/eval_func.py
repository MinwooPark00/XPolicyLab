import torch
import numpy as np

def to_ndarray(sample):
  skip_keys = [
    "rgb_obs", "language_", "frame_count",
    "raw_width","raw_height",
    "raw_w","raw_h",
  ]
  for key in sample.keys():
    if key in skip_keys or "language_" in key:
      continue
    try:
      sample[key] = [v if v is not None else 0 for v in sample[key]]
    except Exception:
      print(key)
      exit()

    sample[key] = [v if v is not None else 0 for v in sample[key]]
    sample[key] = np.array(sample[key])
    # sample[key][~np.isfinite(sample[key])] = 0

EPIC_KITCHEN_HEIGHT = 256
EPIC_KITCHEN_WIDTH = 456


def eval_single_sample(
  raw_data_dict,
  tokenizer,
  model,
  image_width=EPIC_KITCHEN_WIDTH,
  image_height=EPIC_KITCHEN_HEIGHT,
):
  data_dict = {}
  data_dict["images"] = raw_data_dict["image"].to(model.dtype).to(model.device)

  num_images = data_dict["images"].shape[0]

  data_dict["input_ids"] = torch.nn.utils.rnn.pad_sequence(
      [raw_data_dict["input_ids"]], batch_first=True, padding_value=tokenizer.pad_token_id
  ).to(model.device)
  data_dict["attention_mask"]=data_dict["input_ids"].ne(tokenizer.pad_token_id).to(model.device)
  data_dict["labels"] = torch.nn.utils.rnn.pad_sequence(
      [raw_data_dict["labels"]], batch_first=True, padding_value=-100
  ).to(model.device)

  data_dict["inference"] = True
  data_dict["raw_action_labels"] = raw_data_dict["raw_action_label"].to(model.dtype).to(model.device) # No Need to Unsqueeze
  data_dict["raw_action_masks"] = raw_data_dict["raw_action_mask"].to(model.device) # No Need to Unsqueeze

  data_dict["raw_proprio_inputs"] = raw_data_dict["proprio_input"].to(model.dtype).to(model.device)
  data_dict["raw_proprio_inputs_2d"] = raw_data_dict["proprio_input_2d"].to(model.dtype).to(model.device)
  data_dict["raw_proprio_inputs_3d"] = raw_data_dict["proprio_input_3d"].to(model.dtype).to(model.device)
  data_dict["raw_proprio_inputs_rot"] = raw_data_dict["proprio_input_rot"].to(model.dtype).to(model.device)
  data_dict["raw_proprio_inputs_handdof"] = raw_data_dict["proprio_input_handdof"].to(model.dtype).to(model.device)
  data_dict["raw_proprio_inputs_hand_finger_tip"] = raw_data_dict["proprio_input_hand_finger_tip"].to(model.dtype).to(model.device)
  data_dict["raw_ee_movement_masks"] = raw_data_dict["ee_movement_mask"].to(model.dtype).to(model.device)

  with torch.inference_mode():
      # with torch.eval():
      output = model.forward(**data_dict)
  # print(output.prediction)
  return output.prediction.cpu().numpy(), \
    raw_data_dict["raw_image_obs"], \
    raw_data_dict["raw_action_label"].cpu().numpy(), \
    data_dict["raw_action_masks"].cpu().numpy(), \
    output.loss.item()
