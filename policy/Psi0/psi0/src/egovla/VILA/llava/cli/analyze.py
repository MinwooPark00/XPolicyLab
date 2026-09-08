import json
import os
import os.path as osp
import random
from pprint import pprint

import wandb
from datasets import load_dataset
from tqdm import tqdm


def process_ai2d(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-ai2d/ai2d.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_split = info["model_configs"]["test_split"]
    print("ai2d: ", dataset_path, dataset_split)
    data = load_dataset(dataset_path, split=dataset_split)
    print(data)

    items = []
    cols = [
        "doc_id",
        "image",
        "question",
        "target",
        "response",
        "exact_match",
    ]
    tab = wandb.Table(
        columns=cols,
    )

    img_source_list = []
    print("prefetching images")
    for idx, img in enumerate(tqdm(data["image"])):
        img_source_list.append(img)

    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        print(f"Log {idx}:")
        print(_log)
        print("")

        _item = [
            _log["doc_id"],
            # wandb.Image(data["test"][img_index]["image"]),
            wandb.Image(img_source_list[idx]),
            _log["arguments"][0][0],
            _log["target"],
            _log["filtered_resps"][0],
            _log["exact_match"],
        ]
        tab.add_data(*_item)
    return tab


def process_docvqa_val(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-docvqa_val/docvqa_val.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_name = info["model_configs"]["dataset_name"]
    print("docvqa_val: ", dataset_path, dataset_name)
    data = load_dataset(dataset_path, dataset_name, split="validation")
    print(data)

    items = []
    cols = [
        "doc_id",
        "ucsf_document_id",
        "subdomain",
        "image",
        "question",
        "target",
        "response",
        "anls",
    ]
    tab = wandb.Table(
        columns=cols,
    )

    img_source_list = []
    print("prefetching images")
    for idx, img in enumerate(tqdm(data["image"])):
        img_source_list.append(img)

    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # print(_log)
        # print("")

        _item = [
            _log["doc_id"],
            _log["doc"]["ucsf_document_id"],
            _log["doc"]["question_types"][0],
            # wandb.Image(data["test"][img_index]["image"]),
            wandb.Image(img_source_list[idx]),
            _log["doc"]["question"],
            _log["target"],
            _log["filtered_resps"][0],
            _log["anls"],
        ]
        tab.add_data(*_item)
    return tab


def process_chartqa(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-chartqa/chartqa.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_split = info["model_configs"]["test_split"]
    print("chartqa: ", dataset_path, dataset_split)
    data = load_dataset(dataset_path, split=dataset_split)
    print(data)

    items = []
    cols = [
        "doc_id",
        "subdomain",
        "image",
        "question",
        "target",
        "response",
        "relaxed_overall",
    ]
    tab = wandb.Table(
        columns=cols,
    )

    img_source_list = []
    print("prefetching images")
    for idx, img in enumerate(tqdm(data["image"])):
        img_source_list.append(img)

    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # print(_log)
        # print("")
        # exit(0)

        _item = [
            _log["doc_id"],
            _log["doc"]["type"],
            # wandb.Image(data["test"][img_index]["image"]),
            wandb.Image(img_source_list[idx]),
            _log["doc"]["question"],
            _log["doc"]["answer"],
            _log["filtered_resps"][0],
            _log["relaxed_overall"],
        ]
        tab.add_data(*_item)
    return tab


def process_mmmu_val(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-mmmu_val/mmmu_val.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_split = info["model_configs"]["test_split"]
    print("mmmu_val: ", dataset_path, dataset_split)
    data = load_dataset(dataset_path, split=dataset_split)
    print(data)

    items = []
    cols = [
        "doc_id",
        "question_id",
        "subdomain",
        "image",
        "question",
        "target",
        "response",
        "score",
    ]
    tab = wandb.Table(
        columns=cols,
    )

    img_source_list = []
    print("prefetching images")
    for idx, img in enumerate(tqdm(data["image_1"])):
        img_source_list.append(img)

    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # print(_log)
        # print("")

        _item = [
            _log["doc_id"],
            # _log[score_key]["question_id"],
            _log["mmmu_acc"]["id"],
            # _log[score_key]["category"],
            _log["mmmu_acc"]["subdomain"],
            # wandb.Image(data["test"][img_index]["image"]),
            wandb.Image(img_source_list[idx]),
            _log["doc"]["question"],
            str(_log["mmmu_acc"]["answer"]),
            str(_log["mmmu_acc"]["parsed_pred"]),
            str(_log["mmmu_acc"]["answer"]) == str(_log["mmmu_acc"]["parsed_pred"]),
        ]
        tab.add_data(*_item)
    return tab


def process_mme(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-mme/mme.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    data = load_dataset(dataset_path)
    print(data)

    items = []
    cols = [
        "doc_id",
        "question_id",
        "category",
        "image",
        "question",
        "target",
        "response",
        "score",
    ]
    tab = wandb.Table(
        columns=cols,
    )
    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # print(_log)
        # print("")

        img_name = _log["doc"]["question_id"]
        img_index = data["test"]["question_id"].index(img_name)

        resp = _log["resps"][0][0]
        _score = _log["target"].lower() == resp.lower()
        if _score:
            score = "true"
        else:
            score = "false"
        # score = ["no", ]

        score_key = "mme_cognition_score" if "mme_cognition_score" in _log else "mme_percetion_score"

        _item = [
            _log["doc_id"],
            _log[score_key]["question_id"],
            _log[score_key]["category"],
            wandb.Image(data["test"][img_index]["image"]),
            _log["doc"]["question"],
            _log["target"],
            resp,
            score,
        ]
        tab.add_data(*_item)
    return tab


def process_gpq(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-gqa/gqa.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_split = info["model_configs"]["dataset_name"]
    data = load_dataset(dataset_path, dataset_split)
    new_data = load_dataset(dataset_path, "testdev_balanced_images")

    print(data)
    items = []
    cols = [
        "imageId",
        "gqaId",
        "question",
        "image",
        "target",
        "filtered_resps",
        "exact_match",
        "globalGroups",
        "localGroups",
    ]
    tab = wandb.Table(
        columns=cols,
    )
    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # pprint(_log)
        # print("")
        imageId = _log["doc"]["imageId"]
        # print(new_data)
        # print(new_data["testdev"])
        img_index = new_data["testdev"]["id"].index(imageId)
        wandb_image = wandb.Image(new_data["testdev"][img_index]["image"])

        _item = [
            _log["doc"]["imageId"],  # imageId
            _log["doc"]["id"],  # gqaId
            _log["doc"]["question"],  # question
            wandb_image,  # image
            _log["target"],  # ground truth
            _log["filtered_resps"][0],  # filtered_resps
            _log["exact_match"],  # exact_match
            _log["doc"]["groups"]["global"],  # globalGroups
            _log["doc"]["groups"]["local"],  # localGroups
        ]
        tab.add_data(*_item)
    return tab


def process_ocrbench(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-ocrbench/ocrbench.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_split = info["model_configs"]["test_split"]
    print("ocrbench: ", dataset_path, dataset_split)
    data = load_dataset(dataset_path, split=dataset_split)
    print(data)

    items = []
    cols = ["question", "image", "ground_truth", "prediction", "score", "question_type"]
    tab = wandb.Table(
        columns=cols,
    )
    # prefetch images
    img_source_list = []

    # TODO(ligeng): it is weird that prefetching is much faster than loading in the second for-loop.
    print("prefetching images")
    for idx, img in enumerate(tqdm(data["image"])):
        img_source_list.append(img)

    raw_img = data["image"][0]
    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # pprint(_log)
        # print("")
        raw_img = img_source_list[idx]
        wandb_image = wandb.Image(raw_img)

        _item = [
            _log["doc"]["question"],  # question
            wandb_image,  # image
            _log["ocrbench_accuracy"]["ground_truth"][0],  # ground_truth
            _log["ocrbench_accuracy"]["prediction"],  # prediction
            _log["ocrbench_accuracy"]["score"],  # score
            _log["doc"]["question_type"],  # question type
        ]
        tab.add_data(*_item)
    return tab


def process_textvqa(folder, max_samples=20):
    fpath = osp.join(folder, "lmms-textvqa_val/textvqa_val.json")

    with open(fpath) as f:
        info = json.load(f)

    dataset_path = info["model_configs"]["dataset_path"]
    dataset_split = info["model_configs"]["test_split"]
    data = load_dataset(dataset_path, split=dataset_split)
    print(data)

    items = []
    cols = [
        "question_id",
        "question",
        "image",
        "ground_truth",
        "prediction",
        "score",
        "image_classes",
        "image_width",
        "image_height",
    ]
    tab = wandb.Table(
        columns=cols,
    )

    img_source_list = []
    print("prefetching images")
    for idx, img in enumerate(tqdm(data["image"])):
        img_source_list.append(img)

    for idx, _log in enumerate(tqdm(info["logs"])):
        if max_samples > 0 and idx > max_samples:
            break
        # print(f"Log {idx}:")
        # pprint(_log)
        # print("")
        wandb_image = wandb.Image(img_source_list[idx])

        _item = [
            _log["doc"]["question_id"],  # question
            _log["doc"]["question"],  # question
            wandb_image,  # image
            list(set(_log["doc"]["answers"])),  # ground_truth
            _log["filtered_resps"][0],  # prediction
            _log["exact_match"],  # score
            _log["doc"]["image_classes"],  # image classes
            _log["doc"]["image_width"],  # image classes
            _log["doc"]["image_height"],  # image classes
        ]
        tab.add_data(*_item)
    return tab


def main(
    folder="/home/jasonlu/workspace/VILA/runs/eval/qwen2-vl-7b-dynamic-pretrain_sft-20240908141045",
    max_samples=500,
):
    # mname = "yunhaof-july5-llama3_8b-mmc4_spatial_ocr_coyo-stage2-sft-sharegpt4v_sft+vflan+sharegpt_video"
    # folder = osp.join("runs/eval/", mname)
    # folder = "/home/zhijianl/workspace/runs/eval/qwen2-7b-instruct-sft-20240826233532"
    # folder = "/home/jasonlu/workspace/VILA/runs/eval/qwen2-vl-7b-dynamic-pretrain_sft-20240908141045"
    run_name = osp.basename(folder)

    dataset2fn = {
        # "dev": process_ai2d,
        "lmms-chartqa": process_chartqa,
        "lmms-docvqa_val": process_docvqa_val,
        "lmms-mmmu_val": process_mmmu_val,
        "lmms-mme": process_mme,
        "lmms-gqa": process_gpq,
        "lmms-ocrbench": process_ocrbench,
        "lmms-textvqa": process_textvqa,
    }

    for k, v in dataset2fn.items():
        print("collecting data for", k)
        tab = v(folder, max_samples=max_samples)
        if not wandb.run:
            wandb.init(project="vila-ablation", name=run_name)
        wandb.log(
            {
                k: tab,
            }
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
