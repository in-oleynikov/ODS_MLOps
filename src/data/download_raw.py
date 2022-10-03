from datasets import load_dataset
import fire


def save_dataset(path: str):
    raw_datasets = load_dataset("conll2003")
    raw_datasets.save_to_disk(path)


if __name__ == "__main__":
    fire.Fire(save_dataset)
