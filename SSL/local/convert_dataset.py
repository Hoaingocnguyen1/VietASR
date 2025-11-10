import os
import argparse
from datasets import load_dataset, Audio
from tqdm import tqdm
import soundfile as sf


def save_wav_and_trans(dataset, split_dir, dataset_name):
    """
    Save .wav files and text labels into the supervised directory format:
    download/{split}/{dataset_name}/
    """
    dataset_folder = os.path.basename(dataset_name).replace("/", "_")
    data_dir = os.path.join(split_dir, dataset_folder)
    os.makedirs(data_dir, exist_ok=True)

    trans_path = os.path.join(data_dir, "filename.trans.txt")
    with open(trans_path, "w", encoding="utf-8") as f_trans:
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_folder}")):
            # Ensure audio data exists
            if "audio" not in item or item["audio"] is None:
                continue

            audio = item["audio"]
            if audio is None or audio["array"] is None:
                continue

            text = item.get("text", "").strip()
            if not text:
                continue

            audio_filename = f"{dataset_folder}_{i:06d}.wav"
            audio_path = os.path.join(data_dir, audio_filename)

            # Save audio
            sf.write(audio_path, audio["array"], audio["sampling_rate"])

            # Save transcript line
            f_trans.write(f"{os.path.splitext(audio_filename)[0]} {text}\n")


def convert_hf_to_supervised(dataset_name, output_dir):
    """
    Convert a Hugging Face dataset into ASR supervised structure (non-streaming).
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print("Dataset splits:", list(dataset.keys()))

    for split in dataset.keys():
        print(f"\nConverting split: {split}")
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        dataset_split = dataset[split].cast_column("audio", Audio())
        save_wav_and_trans(dataset_split, split_dir, dataset_name)


def convert_hf_to_supervised_streaming(dataset_name, output_dir):
    """
    Convert dataset in streaming mode (for large datasets).
    """
    print(f"Loading dataset (streaming): {dataset_name}")
    dataset = load_dataset(dataset_name, streaming=True)
    print("Dataset splits:", list(dataset.keys()))

    for split in dataset.keys():
        print(f"\nConverting split: {split}")
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Convert each example on the fly
        trans_path = os.path.join(
            split_dir, os.path.basename(dataset_name).replace("/", "_"), "filename.trans.txt"
        )
        data_dir = os.path.dirname(trans_path)
        os.makedirs(data_dir, exist_ok=True)

        with open(trans_path, "w", encoding="utf-8") as f_trans:
            for i, item in enumerate(tqdm(dataset[split], desc=f"Streaming {split}")):
                audio = item["audio"]
                if audio is None or audio["array"] is None:
                    continue

                text = item.get("text", "").strip()
                if not text:
                    continue

                audio_filename = f"{os.path.basename(dataset_name).replace('/', '_')}_{i:06d}.wav"
                audio_path = os.path.join(data_dir, audio_filename)
                sf.write(audio_path, audio["array"], audio["sampling_rate"])

                f_trans.write(f"{os.path.splitext(audio_filename)[0]} {text}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert HF dataset to ASR supervised format")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name, e.g. doof-ferb/vlsp2020_vinai_100h")
    parser.add_argument("--output", default="download", help="Root output directory (default: download)")
    parser.add_argument("--mode", choices=["full", "stream"], default="full", help="Conversion mode: full or stream")
    args = parser.parse_args()

    if args.mode == "stream":
        convert_hf_to_supervised_streaming(args.dataset, args.output)
    else:
        convert_hf_to_supervised(args.dataset, args.output)

    print("\n Conversion finished successfully!")
    print(f"Data saved under: {os.path.join(args.output)}")


if __name__ == "__main__":
    main()
