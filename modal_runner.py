import json
import os
import pathlib
import subprocess
from typing import Optional

import modal

# -----------------------------
# Modal app + infrastructure
# -----------------------------
APP_NAME = "bge-m3-finetune-modal"
GPU_TYPE = "RTX-PRO-6000" # Requested explicitly per your requirement.

RAW_DATA_VOLUME_NAME = "bge-m3-finetune-dataset-raw"
EXTRACTED_DATA_VOLUME_NAME = "bge-m3-finetune-dataset-extracted"
OUTPUT_VOLUME_NAME = "bge-m3-finetune-final"

RAW_DATA_MOUNT = pathlib.Path("/vol/raw")
EXTRACTED_DATA_MOUNT = pathlib.Path("/vol/extracted")
OUTPUT_MOUNT = pathlib.Path("/vol/output")

# Kaggle dataset requested:
# https://www.kaggle.com/datasets/aksj4in/bge-m3-finetune-dataset/
KAGGLE_DATASET = "aksj4in/bge-m3-finetune-dataset"

# Intentionally plain env vars (not Modal Secret), as requested.
# Set either KAGGLE_API_TOKEN or the legacy KAGGLE_USERNAME/KAGGLE_KEY pair before running.
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "wget", "tar", "clang", "build-essential")
    .pip_install(
        "kaggle",
        "datasets",
        "accelerate",
        "deepspeed",
        "bitsandbytes",
        "pytrec_eval",
        "faiss-gpu-cu12",
        "transformers",
    )
)

app = modal.App(APP_NAME)

raw_data_vol = modal.Volume.from_name(RAW_DATA_VOLUME_NAME, create_if_missing=True)
extracted_data_vol = modal.Volume.from_name(
    EXTRACTED_DATA_VOLUME_NAME, create_if_missing=True
)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _run(cmd: str, cwd: Optional[pathlib.Path] = None) -> None:
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd) if cwd else None)


def _write_ds_config(path: pathlib.Path) -> pathlib.Path:
    ds_config = {
        "train_batch_size": "auto",
        # "train_micro_batch_size_per_gpu": "auto",
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 1},
    }
    with path.open("w") as f:
        json.dump(ds_config, f)
    return path


# def _find_train_jsonl(dataset_root: pathlib.Path) -> pathlib.Path:
#     preferred = list(dataset_root.rglob("miracl_en_len-0-500.jsonl"))
#     if preferred:
#         return preferred[0]

#     jsonls = sorted(dataset_root.rglob("*.jsonl"))
#     if not jsonls:
#         raise FileNotFoundError(
#             f"No .jsonl training file found under {dataset_root}. Check Kaggle dataset contents."
#         )
#     return jsonls[0]
    
# def _find_train_jsonls_bengali(
#     dataset_root: pathlib.Path,
#     max_total_bytes: int = 5 * 1024 ** 3,  # 5 GB
# ) -> list[pathlib.Path]:
#     """
#     Collect Bengali (bn) MIRACL jsonl files up to max_total_bytes total.
#     Files are added in sorted order (shortest/smallest range first) until
#     the 5 GB cap is reached.
#     """
#     bn_files = sorted(dataset_root.rglob("miracl_bn_*.jsonl")) + sorted(dataset_root.rglob("miracl_te_*.jsonl")) + sorted(dataset_root.rglob("miracl_sw_*.jsonl")) + sorted(dataset_root.rglob("miracl_hi_*.jsonl"))
    

#     if not bn_files:
#         raise FileNotFoundError(
#             f"No Bengali .jsonl files found under {dataset_root}. "
#             "Check that the Kaggle dataset was extracted correctly."
#         )

#     selected = []
#     total_bytes = 0

#     for f in bn_files:
#         file_size = f.stat().st_size
#         if total_bytes + file_size > max_total_bytes:
#             print(
#                 f"  [SKIP] {f.name} ({file_size / 1024**2:.1f} MB) — "
#                 f"would exceed 5 GB cap. Stopping here."
#             )
#             break
#         selected.append(f)
#         total_bytes += file_size
#         print(f"  [ADD]  {f.name} ({file_size / 1024**2:.1f} MB) — "
#               f"running total: {total_bytes / 1024**3:.2f} GB")

#     if not selected:
#         raise RuntimeError(
#             "No files could be selected — even the first file exceeds 5 GB."
#         )

#     print(f"\nSelected {len(selected)} files, "
#           f"total size: {total_bytes / 1024**3:.2f} GB")
#     return selected


def _find_train_jsonls_bengali(
    dataset_root: pathlib.Path,
    max_total_bytes: int = 5 * 1024 ** 3,  # 5 GB
) -> pathlib.Path:
    """
    Collect MIRACL jsonl files up to max_total_bytes total.
    Files are added in sorted order (shortest/smallest range first) until
    the 5 GB cap is reached. Copies them into a directory and returns that directory path.
    """
    import shutil
    
    partial_dir = dataset_root / "bge-m3-data" / "MIRACL-PARTIAL"
    
    # Check if the directory already exists and is populated
    if partial_dir.exists() and any(partial_dir.iterdir()):
        print(f"\n[INFO] Directory {partial_dir} already exists. Reusing existing files.")
        return partial_dir

    bn_files = (
        sorted(dataset_root.rglob("miracl_bn_*.jsonl")) + 
        sorted(dataset_root.rglob("miracl_te_*.jsonl")) + 
        sorted(dataset_root.rglob("miracl_sw_*.jsonl")) + 
        sorted(dataset_root.rglob("miracl_hi_*.jsonl")) +
        sorted(dataset_root.rglob("miracl_ar_*.jsonl"))
    )

    if not bn_files:
        raise FileNotFoundError(
            f"No matching .jsonl files found under {dataset_root}. "
            "Check that the Kaggle dataset was extracted correctly."
        )

    # Create the directory
    partial_dir.mkdir(parents=True, exist_ok=True)
    
    selected_count = 0
    total_bytes = 0

    for f in bn_files:
        file_size = f.stat().st_size
        if total_bytes + file_size > max_total_bytes:
            print(
                f"  [SKIP] {f.name} ({file_size / 1024**2:.1f} MB) — "
                f"would exceed 5 GB cap. Stopping here."
            )
            break
        
        # Copy the file to the new directory
        dest_file = partial_dir / f.name
        shutil.copy2(f, dest_file)
        
        selected_count += 1
        total_bytes += file_size
        print(f"  [COPIED]  {f.name} ({file_size / 1024**2:.1f} MB) — "
              f"running total: {total_bytes / 1024**3:.2f} GB")

    if selected_count == 0:
        raise RuntimeError(
            "No files could be selected — even the first file exceeds 5 GB."
        )

    print(f"\nSelected and copied {selected_count} files into {partial_dir}, "
          f"total size: {total_bytes / 1024**3:.2f} GB")
          
    return partial_dir


def _download_kaggle_zip(raw_root: pathlib.Path) -> pathlib.Path:
    marker = raw_root / ".download_complete"
    if marker.exists():
        print(f"Dataset zip already present at {raw_root}, skipping re-download.")
        zips = sorted(raw_root.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(
                f"Marker exists but no zip file found in {raw_root}."
            )
        return zips[0]

    if not os.environ.get("KAGGLE_API_TOKEN") and (
        not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY")
    ):
        raise RuntimeError(
            "Set KAGGLE_API_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY, before running."
        )

    raw_root.mkdir(parents=True, exist_ok=True)
    _run(f"kaggle datasets download -d {KAGGLE_DATASET} -p {raw_root}")
    marker.write_text("ok\n")
    raw_data_vol.commit()

    zips = sorted(raw_root.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"Kaggle download completed, but no zip file found in {raw_root}."
        )
    return zips[0]


def _extract_kaggle_zip(zip_path: pathlib.Path, extracted_root: pathlib.Path) -> None:
    marker = extracted_root / ".extract_complete"
    if marker.exists():
        print(f"Dataset already extracted at {extracted_root}, skipping unzip.")
        return

    extracted_root.mkdir(parents=True, exist_ok=True)
    _run(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import zipfile\n"
        f"zip_path = Path(r'{zip_path}')\n"
        f"extract_root = Path(r'{extracted_root}')\n"
        "with zipfile.ZipFile(zip_path, 'r') as archive:\n"
        "    archive.extractall(extract_root)\n"
        "print(f'Extracted {zip_path} -> {extract_root}')\n"
        "PY"
    )
    marker.write_text("ok\n")
    extracted_data_vol.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=7 * 60 * 60,
    volumes={
        str(RAW_DATA_MOUNT): raw_data_vol,
        str(EXTRACTED_DATA_MOUNT): extracted_data_vol,
        str(OUTPUT_MOUNT): output_vol,
    },
    env={
        "KAGGLE_API_TOKEN": KAGGLE_API_TOKEN,
        "KAGGLE_USERNAME": KAGGLE_USERNAME,
        "KAGGLE_KEY": KAGGLE_KEY,
        "HF_HOME": str(EXTRACTED_DATA_MOUNT / "cache" / "hf"),
    },
)
def run_pipeline() -> None:
    raw_root = RAW_DATA_MOUNT / "dataset"
    extracted_root = EXTRACTED_DATA_MOUNT / "dataset"
    cache_model = EXTRACTED_DATA_MOUNT / "cache" / "model"
    cache_data = EXTRACTED_DATA_MOUNT / "cache" / "data"

    output_model_dir = OUTPUT_MOUNT / "bge-m3-finetuned"
    eval_dir = OUTPUT_MOUNT / "miracl_eval"

    raw_root.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    cache_model.mkdir(parents=True, exist_ok=True)
    cache_data.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Notebook cell equivalent: !nvcc --version
    # _run("nvcc --version")

    # Notebook cell equivalent (in image build here):
    # !pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
    # !pip install FlagEmbedding[finetune] datasets accelerate deepspeed bitsandbytes
    _run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    )

    # _run(
    #     "pip install 'FlagEmbedding[finetune]' datasets accelerate deepspeed bitsandbytes"
    # )

    _run(
        "pip install 'FlagEmbedding[finetune]@git+https://github.com/AkshajBPGC/FlagEmbedding.git' datasets accelerate deepspeed bitsandbytes"
    )

    # Notebook commented cell equivalents preserved:
    # !git clone https://github.com/FlagOpen/FlagEmbedding.git
    # %cd FlagEmbedding
    # !pip install .[finetune]

    # Notebook commented cell equivalents preserved:
    # from datasets import load_dataset
    # dataset = load_dataset("Shitao/bge-m3-data")
    # print(dataset)
    # print(dataset.keys())
    # !wget https://huggingface.co/datasets/Shitao/bge-m3-data/resolve/main/bge-m3-data.tar.gz

    # Notebook commented cell equivalent preserved:
    # !tar -tf bge-m3-data.tar.gz

    # Replaced source dataset with Kaggle as requested.
    zip_path = _download_kaggle_zip(raw_root)
    _extract_kaggle_zip(zip_path, extracted_root)

    #train_data = _find_train_jsonl(extracted_root)
    #print(f"Using train file: {train_data}")
    
    train_files = _find_train_jsonls_bengali(extracted_root)
    train_data = train_files
    # train_data = " ".join(str(f) for f in train_files)
    # print(f"Using {len(train_files)} Bengali training files")

    ds_config_path = OUTPUT_MOUNT / "ds_config.json"
    _write_ds_config(ds_config_path)


    print(f"Train dir: {train_data}")

    # Notebook training cell equivalent.
    _run(
        "torchrun --nproc_per_node 1 "
        "-m FlagEmbedding.finetune.embedder.encoder_only.m3 "
        "--model_name_or_path BAAI/bge-m3-unsupervised "
        f"--train_data {train_data} "
        f"--cache_dir {cache_model} "
        f"--cache_path {cache_data} "
        "--optim adamw_bnb_8bit "
        "--train_group_size 4 "
        "--query_max_len 512 "
        "--passage_max_len 512 "
        "--pad_to_multiple_of 8 "
        "--knowledge_distillation True "
        "--same_dataset_within_batch True "
        "--small_threshold 0 "
        "--drop_threshold 0 "
        f"--output_dir {output_model_dir} "
        "--learning_rate 1e-5 "
        "--fp16 "
        "--num_train_epochs 50 "
        "--per_device_train_batch_size 24 "
        "--gradient_accumulation_steps 8 "
        "--dataloader_drop_last True "
        "--warmup_steps 100 "
        # "--sub_batch_size 2 "
        "--gradient_checkpointing "
        f"--deepspeed {ds_config_path} "
        "--logging_steps 10 "
        "--save_steps 100 "
        "--negatives_cross_device "
        "--temperature 0.02 "
        "--sentence_pooling_method cls "
        "--normalize_embeddings True "
        "--kd_loss_type m3_kd_loss "
        "--unified_finetuning True "
        "--use_self_distill True "
        "--fix_encoder False "
        "--self_distill_start_step 0"
    )


    # Notebook tokenizer-save cell equivalent.
    # _run(
    #     "python - <<'PY'\n"
    #     "from pathlib import Path\n"
    #     "from transformers import AutoTokenizer\n"
    #     "tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')\n"
    #     "ckpt = Path('/vol/output/bge-m3-finetuned/checkpoint-299')\n"
    #     "ckpt.mkdir(parents=True, exist_ok=True)\n"
    #     "tokenizer.save_pretrained(str(ckpt))\n"
    #     "print('Tokenizer saved! Your fine-tuned model is fully ready to use.')\n"
    #     "PY"
    # )


    # Notebook BENCHMARKING cell equivalent.
    _run("pip install pytrec_eval faiss_gpu_cu12 datasets==2.18.0")
    # !pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

    # # Benchmark the already finetuned model.
    # _run(
    #     "python -m FlagEmbedding.evaluation.miracl "
    #     "--eval_name miracl "
    #     "--dataset_dir ./miracl_data "
    #     "--dataset_names sw bn "
    #     "--splits dev "
    #     f"--corpus_embd_save_dir {eval_dir / 'corpus_embd_ft'} "
    #     f"--output_dir {eval_dir / 'search_results_ft'} "
    #     "--search_top_k 1000 "
    #     "--rerank_top_k 100 "
    #     f"--cache_path {cache_data} "
    #     "--overwrite True "
    #     "--k_values 10 100 "
    #     "--eval_output_method markdown "
    #     f"--eval_output_path {eval_dir / 'results_ft.md'} "
    #     "--eval_metrics ndcg_at_10 recall_at_100 "
    #     f"--embedder_name_or_path {output_model_dir / 'checkpoint-680'} "
    #     "--embedder_model_class encoder-only-m3 "
    #     "--devices cuda:0 "
    #     f"--cache_dir {cache_model} "
    #     "--reranker_query_max_length 512 "
    #     "--reranker_max_length 1024"
    # )


    # # Benchmark the unsupervised model.
    # _run(
    #     "python -m FlagEmbedding.evaluation.miracl "
    #     "--eval_name miracl "
    #     "--dataset_dir ./miracl_data "
    #     "--dataset_names sw "
    #     "--splits dev "
    #     f"--corpus_embd_save_dir {eval_dir / 'corpus_embd_unsup'} "
    #     f"--output_dir {eval_dir / 'search_results_unsup'} "
    #     "--search_top_k 1000 "
    #     "--rerank_top_k 100 "
    #     f"--cache_path {cache_data} "
    #     "--overwrite True "
    #     "--k_values 10 100 "
    #     "--eval_output_method markdown "
    #     f"--eval_output_path {eval_dir / 'results_unsup.md'} "
    #     "--eval_metrics ndcg_at_10 recall_at_100 "
    #     "--embedder_name_or_path BAAI/bge-m3-unsupervised "
    #     "--embedder_model_class encoder-only-m3 "
    #     "--devices cuda:0 "
    #     f"--cache_dir {cache_model} "
    #     "--reranker_query_max_length 512 "
    #     "--reranker_max_length 1024"
    # )

    # Persist all artifacts/checkpoints/eval outputs.
    raw_data_vol.commit()
    extracted_data_vol.commit()
    output_vol.commit()


@app.local_entrypoint()
def main() -> None:
    run_pipeline.remote()
