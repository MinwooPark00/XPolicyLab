#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import VLAConsumerDataset
from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone
from models.hrdt_runner import HRDTRunner


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix == ".safetensors":
        import safetensors.torch

        missing, unexpected = safetensors.torch.load_model(model, checkpoint_path, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        return
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = None
        for key in ("state_dict", "model", "module"):
            if isinstance(ckpt, dict) and key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            state = ckpt
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default="configs/hrdt_finetune_lerobot.yaml")
    parser.add_argument("--vision_encoder", default="dino-siglip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    vision_encoder = DinoSigLIPViTBackbone(
        vision_backbone_id=args.vision_encoder,
        image_resize_strategy="letterbox"
        if config["dataset"]["image_aspect_ratio"] == "pad"
        else "resize-naive",
        default_image_size=384,
    )
    vision_encoder.eval()
    image_transform = vision_encoder.get_image_transform()

    dataset = VLAConsumerDataset(
        config=config,
        image_transform=image_transform,
        num_cameras=config["common"]["num_cameras"],
        dataset_type="finetune",
        dataset_name="lerobot",
        dataset_root=args.data_root,
        use_precomp_lang_embed=True,
        upsample_rate=1,
        val=False,
    )

    hrdt = HRDTRunner(
        state_dim=config["common"]["state_dim"],
        action_dim=config["common"]["action_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        act_pos_emb_config=[("state", 1), ("action", config["common"]["action_chunk_size"])],
        img_pos_emb_config=[
            ("image", (config["common"]["img_history_size"], config["common"]["num_cameras"], -vision_encoder.num_patches)),
        ],
        lang_pos_emb_config=[("language", -config["dataset"]["tokenizer_max_length"])],
        max_img_len=config["common"]["img_history_size"] * config["common"]["num_cameras"] * vision_encoder.num_patches,
        max_lang_len=config["dataset"]["tokenizer_max_length"],
        training_mode="lang",
        mode="finetune",
        dtype=torch.bfloat16,
    )
    print("Loading checkpoint...")
    load_checkpoint(hrdt, Path(args.checkpoint_path))
    print("Moving models to device...")
    vision_encoder.to(device, dtype=torch.bfloat16)
    hrdt.to(device, dtype=torch.bfloat16)
    print("Starting evaluation...")
    hrdt.eval()

    total_abs = None
    total_count = 0

    with torch.no_grad():
        num_samples = min(args.num_samples, len(dataset))
        for idx in range(num_samples):
            sample = dataset[idx]
            print(f"sample={idx:03d} loaded")

            images = {
                key: value.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                for key, value in sample["images"].items()
            }
            image_key = next(iter(images))
            batch_size, _, c, h, w = images[image_key].shape
            for key in images:
                images[key] = images[key].view(-1, c, h, w)
            print(f"sample={idx:03d} encoding_images")
            image_features = vision_encoder(images).detach()
            image_features = image_features.view((batch_size, -1, vision_encoder.embed_dim))
            print(f"sample={idx:03d} predicting_actions")

            pred_actions = hrdt.predict_action(
                state_tokens=sample["states"].unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                image_tokens=image_features,
                lang_tokens=sample["lang_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                lang_attn_mask=torch.ones(
                    (1, sample["lang_embeds"].shape[0]),
                    device=device,
                    dtype=torch.bool,
                ),
            ).to(dtype=torch.float32)

            gt_actions = sample["actions"].unsqueeze(0).to(dtype=torch.float32)
            min_len = min(pred_actions.shape[1], gt_actions.shape[1])
            min_dim = min(pred_actions.shape[2], gt_actions.shape[2])
            diff = torch.abs(pred_actions[:, :min_len, :min_dim] - gt_actions[:, :min_len, :min_dim]).sum(dim=(0, 1))
            total_abs = diff if total_abs is None else total_abs + diff
            total_count += min_len
            print(f"sample={idx:03d} mean_l1={diff.mean().item() / min_len:.6f}")

    if total_abs is None or total_count == 0:
        raise RuntimeError("No samples were evaluated.")

    avg_l1 = total_abs / total_count
    print("avg_l1_per_dim=", avg_l1.tolist())
    print("avg_l1_mean=", avg_l1.mean().item())


if __name__ == "__main__":
    main()
