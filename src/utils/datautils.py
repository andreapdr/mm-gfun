import os
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict

DATASET_URL = os.environ.get("DATASET_URL", "https://zenodo.org/record/7326406/files/GLAMI-1M-dataset.zip?download=1")
EXTRACT_DIR = os.environ.get("EXTRACT_DIR", "/tmp/GLAMI-1M")
DATASET_SUBDIR = "GLAMI-1M-dataset"
DATASET_DIR = dataset_dir = EXTRACT_DIR + "/" + DATASET_SUBDIR
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/GLAMI-1M/models")
EMBS_DIR = EXTRACT_DIR + "/embs"
CLIP_VISUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-visual"
CLIP_TEXTUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-textual"
# CLIP_VISUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-l5b-visual"
# CLIP_TEXTUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-l5b-textual"
CLIP_EN_TEXTUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-en-textual"
GENERATED_DIR = EXTRACT_DIR + "/generated_images"

COL_NAME_ITEM_ID = "item_id"
COL_NAME_IMAGE_ID = "image_id"
COL_NAME_IMAGE_FILE = "image_file"
COL_NAME_IMAGE_URL = "image_url"
COL_NAME_NAME = "name"
COL_NAME_DESCRIPTION = "description"
COL_NAME_GEO = "geo"
COL_NAME_CATEGORY = "category"
COL_NAME_CAT_NAME = "category_name"
COL_NAME_LABEL_SOURCE = "label_source"
COL_NAME_EMB_FILE = "emb_file"
COL_NAME_MASK_FILE = "mask_file"
DEFAULT_IMAGE_SIZE = (298, 228)

COUNTRY_CODE_TO_COUNTRY_NAME = {
    "cz": "Czechia",
    "sk": "Slovakia",
    "ro": "Romania",
    "gr": "Greece",
    "si": "Slovenia",
    "hu": "Hungary",
    "hr": "Croatia",
    "es": "Spain",
    "lt": "Lithuania",
    "lv": "Latvia",
    "tr": "Turkey",
    "ee": "Estonia",
    "bg": "Bulgaria",
}

COUNTRY_CODE_TO_COUNTRY_NAME_W_CC = {name + f' ({cc})' for cc, name in COUNTRY_CODE_TO_COUNTRY_NAME}


def get_glami_dataframe(split_type: str, dataset_dir=DATASET_DIR):
    assert split_type in ("train", "test")
    df = pd.read_csv(dataset_dir + f"/GLAMI-1M-{split_type}.csv")
    df[COL_NAME_IMAGE_FILE] = dataset_dir + "/images/" + df[COL_NAME_IMAGE_ID].astype(str) + ".jpg"
    df[COL_NAME_DESCRIPTION] = df[COL_NAME_DESCRIPTION].fillna('')
    assert os.path.exists(df.loc[0, COL_NAME_IMAGE_FILE])
    return df[[COL_NAME_ITEM_ID, COL_NAME_IMAGE_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_GEO, COL_NAME_CATEGORY, COL_NAME_CAT_NAME, COL_NAME_LABEL_SOURCE, COL_NAME_IMAGE_FILE]]


def load_glami(split, datadir):
    df = get_glami_dataframe(split, datadir)
    text = (df["name"] + " " + df["description"]).to_list()
    img_paths = df.image_file.to_list()
    category = df.category_name.to_list()
    metadata = df.to_dict()
    return text, img_paths, category, metadata


def convert_glami_to_hf(datadir="/media/datasets/GLAMI-1M-dataset"):
    dataset = DatasetDict()

    df = get_glami_dataframe(split_type="train", dataset_dir=str(datadir))
    df_te = get_glami_dataframe(split_type="test", dataset_dir=str(datadir))

    df["text"] = df["name"] + " " + df["description"]
    df_te["text"] = df_te["name"] + " " + df_te["description"]

    dataset_tr = Dataset.from_pandas(df)         # import dataset (hf) from pandas dataframe
    dataset_te = Dataset.from_pandas(df_te)
    dataset["train"] = dataset_tr 
    dataset["original_test"] = dataset_te
    dataset = dataset.rename_column("image_file", "image")    

    dataset.update(dataset["train"].train_test_split(train_size=0.8, seed=42))
    dataset["validation"] = dataset.pop("test")
    dataset["test"] = dataset.pop("original_test")
    
    label_encoder = LabelEncoder(label_mapper_file=str(Path.joinpath(Path("~/datasets/GLAMI-1M-dataset").expanduser(), "cat_mapper.csv")))
    dataset = dataset.map(label_encoder, batched=True, batch_size=256, num_proc=4)
    
    dataset.save_to_disk(Path("/media/datasets/hf_datasets/GLAMI-1M-dataset"))


class LabelEncoder:
    def __init__(self, label_mapper_file):
        mapper = pd.read_csv(label_mapper_file)
        self.idx2label = {row.label_id: row.category_name for _, row in mapper.iterrows()}
        self.label2idx = {v: k for k, v in self.idx2label.items()}
    
    def __call__(self, features, column_name="category_name"):
        if isinstance(features[column_name], list):
            features["labels"] = [self.label2idx[l] for l in features[column_name]]
        else:
            raise NotImplementedError
        return features



def create_glami_mapper():
    GLAMI_DIR = "~/datasets/GLAMI-1M-dataset"
    datadir = Path(GLAMI_DIR).expanduser()
    df = get_glami_dataframe(split_type="train", dataset_dir=str(datadir))
    
    # saving mapper category_name -> category_id -> label_id
    mapper = []
    cat_names = df.category_name.unique()
    cat_ids = df.cateogory.unique()
    for i, (cat_name, cat_id) in enumerate(zip(cat_names, cat_ids)):
        mapper.append([cat_name, cat_id, i])
    mapper_df = pd.DataFrame(mapper, columns=["category_name", "category", "label_id"])
    mapper_df.to_csv(Path.joinpath(datadir, "cat_mapper.csv"), index=False)


if __name__ == "__main__":
    convert_glami_to_hf()