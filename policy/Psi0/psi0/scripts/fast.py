import argparse
import torch
import numpy as np
import tqdm, time
import os
import shutil
from transformers import AutoProcessor

def copy_processing_action_tokenizer(pretrained_path):
    src = os.path.join(os.path.dirname(__file__), '../src/fast/pi/processing_action_tokenizer.py')
    dst = os.path.join(pretrained_path, 'processing_action_tokenizer.py')
    shutil.copyfile(src, dst)
    print(f"Copied processing_action_tokenizer.py to {dst}")

def fit(train_dataset, num_action_chunks: int, time_horizon: int = 16, action_dim: int = 48, scale: int = 10, vocab_size: int = 2048):
    import glob
    batch_files = glob.glob("src/fast/batch_actions_normed_*.npy")
    if not batch_files:
        all_actions = []
        idx = 0
        # NOTE: EgoDex has wrong dataset length: it returns number of episodes instead of number of action chunks
        for data in tqdm.tqdm(train_dataset, total=num_action_chunks):
            idx += 1
            if idx >= num_action_chunks:
                break
            all_actions.append(data["raw_actions"])

        batch_actions_normed = np.stack(all_actions) # (N, T, D)
        np.save(f"src/fast/batch_actions_normed_{time.time()}.npy", batch_actions_normed)
    else:
        print("loading existing action chunks...")
        batch_list = []
        for file in batch_files:
            arr = np.load(file)
            arr = arr[:, :time_horizon, :]
            batch_list.append(arr)
        batch_actions_normed = np.concatenate(batch_list, axis=0)
        print("loaded", batch_actions_normed.shape)
    
    tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    tokenizer = tokenizer.fit(
        batch_actions_normed, 
        scale=scale, 
        vocab_size=vocab_size, 
        time_horizon=time_horizon, 
        action_dim=action_dim
    )
    pretrained_path = f"src/fast/egodex-rel-50w-{time_horizon}x{action_dim}-v{vocab_size}-s{scale}"
    tokenizer.save_pretrained(pretrained_path)
    print("Tokenizer saved.")

    # hack processing_action_tokenizer.py to the pretrained_path
    print("hacking the decoding ...")
    copy_processing_action_tokenizer(pretrained_path)
    
    return pretrained_path

def validate(tokenizer, val_dataset, action_max, action_min, add_token_noise: bool = True):
    errors = []
    exceptions = []
    token_lens = []
    min_token = np.inf
    max_token = -np.inf
    for _ in range(100):
        rand_idx = np.random.randint(0, val_dataset.dataset_length) # _
        batch = val_dataset[rand_idx]["raw_actions"]

        action_tokens = tokenizer(batch)
        token_lens.append(len(action_tokens[0]))

        noisy_tokens = list(action_tokens[0])

        if add_token_noise:
            num_to_perturb = max(1, int(len(noisy_tokens) * 0.2))
            if True: # pertrub tail tokens only
                lower_bound = len(action_tokens[0]) - num_to_perturb
                pert_indices = np.random.choice(len(noisy_tokens)-lower_bound, size=num_to_perturb, replace=False)
                pert_indices = pert_indices + lower_bound
            else:
                pert_indices = np.random.choice(num_to_perturb, size=num_to_perturb, replace=False)

            noises = []
            for idx in pert_indices:
                noise = int(np.random.randint(-10, 10))
                noises.append(noise)
                noisy_tokens[idx] = int(noisy_tokens[idx]) + noise

        # Expose noisy tokens as `fake_tokens` so the downstream decode attempt can use them
        fake_tokens = noisy_tokens
        # print(f"min token id: {min(fake_tokens)}, max token id: {max(fake_tokens)}")
        # assert min(action_tokens[0]) >= tokenizer.min_token and max(action_tokens[0]) <= tokenizer.vocab_size - 1
        min_token = min(min_token, min(action_tokens[0]))
        max_token = max(max_token, max(action_tokens[0]))
        try:
            fake_action = tokenizer.decode([fake_tokens])[0]
            err = np.abs(fake_action-batch).mean(0)
            err = np.clip(err, 0, 1) * (action_max - action_min)

            errors.append(err.mean())
            exceptions.append((err>0.1).sum())

        except Exception:
            print("failed")

    print("avg error:", np.mean(errors))
    print("number of outliers (>0.1)" , sum(exceptions))
    print("avg token len:", np.mean(token_lens))
    print("max token len:", np.max(token_lens))
    print("min token len:", np.min(token_lens))
    print(f"overall min token id: {min_token}, max token id: {max_token}")


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv(verbose=True)

    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser(description='Calculate EgoDex dataset statistics')
    parser.add_argument('--training_script', type=str,
                       default="scripts/train/psi0/pretrain-egodex-psi0-fast.sh")
    parser.add_argument('--num_action_chunks', type=int, default=100_000,
                        help='Number of action chunks to use for fitting the tokenizer')
    parser.add_argument('--scale', type=int, default=10,
                        help='Scale for DCT coefficient quantization')
    parser.add_argument('--vocab-size', type=int, default=2048,
                        help='vocabulary size for the fast tokenizer')
    
    args = parser.parse_args()
    train_script = args.training_script
    num_action_chunks = args.num_action_chunks
    scale = args.scale
    vocab_size = args.vocab_size
    action_dim = 48

    from psi.utils import parse_args_to_tyro_config, seed_everything
    from psi.config.config import LaunchConfig

    config:LaunchConfig = parse_args_to_tyro_config(train_script) # type: ignore
    seed_everything(config.seed or 42)

    from psi.config.data_lerobot import LerobotDataConfig
    data_cfg: LerobotDataConfig = config.data # type: ignore

    from psi.config.model_qwen3vl import Qwen3VLModelConfig
    model_cfg: Qwen3VLModelConfig = config.model # type: ignore

    vlm_processor = AutoProcessor.from_pretrained(model_cfg.model_name_or_path)
    # tokenizer = vlm_processor.tokenizer

    transform_kwargs= {}
    train_dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)
    val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)

    print("Dataset lengths:", train_dataset.dataset_length, val_dataset.dataset_length)

    assert (hasattr(data_cfg, "use_delta_actions") and data_cfg.use_delta_actions) or \
         (hasattr(data_cfg.transform.repack, "use_delta_actions") and data_cfg.transform.repack.use_delta_actions), \
        "Delta actions should be used when doing pretraining."

    """ actions = []
    for frame in tqdm.tqdm(train_dataset, total=train_dataset.dataset_length):
        actions.append(frame["raw_actions"])
    np.save("all_train_actions.npy", np.concatenate(actions, axis=0)) """

    maxmin = data_cfg.transform.field
    action_min = np.array(maxmin.action_min, dtype=np.float32)
    action_max = np.array(maxmin.action_max, dtype=np.float32)

    original_tokenizer = AutoProcessor.from_pretrained("src/fast/pi", trust_remote_code=True)
    print("\nTest: original fast tokenizer, reconstruction without noise")
    validate(original_tokenizer, val_dataset, action_max, action_min, add_token_noise=False)
    print("\nTest: original fast tokenizer, reconstruction with noise")
    validate(original_tokenizer, val_dataset, action_max, action_min, add_token_noise=True)

    pretrained_path = f"src/fast/egodex-rel-50w-{data_cfg.chunk_size}x{action_dim}-v{vocab_size}-s{scale}"
    if not os.path.exists(pretrained_path):
        print("Fitting new tokenizer...")
        saved_path = fit(
            train_dataset, 
            num_action_chunks, 
            time_horizon=data_cfg.chunk_size, 
            action_dim=action_dim, 
            scale=scale, 
            vocab_size=vocab_size
        )

    new_tokenizer = AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)

    print("\nTest: trained tokenizer, reconstruction (train) without noise")
    validate(new_tokenizer, train_dataset, action_max, action_min, add_token_noise=False)
    print("\nTest: trained tokenizer, reconstruction (train) with noise")
    validate(new_tokenizer, train_dataset, action_max, action_min, add_token_noise=True)

    print("\nTest: trained tokenizer, reconstruction (val) without noise")
    validate(new_tokenizer, val_dataset, action_max, action_min, add_token_noise=False)
    print("\nTest: trained tokenizer, reconstruction (val) with noise")
    validate(new_tokenizer, val_dataset, action_max, action_min, add_token_noise=True)