# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import numpy as np
import json
import argparse
import pandas as pd

from os.path import join
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_file_path",
    type=str,
    default="./data_info.csv",
    help="Path to `data.info.csv` file.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./",
    help="Path to save the csv/json files for training.",
)
parser.add_argument(
    "--test", action="store_true", help="test flag, used to generate test filelist"
)

SEED_NUM = 2022
np.random.seed(SEED_NUM)


def create_splits(data_path, save_path):

    data_info = np.loadtxt(str(data_path), dtype=str, delimiter=",")

    split2files = defaultdict(list)
    heads = [
        "id",
        "speaker",
        "type",
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]
    for i, x in enumerate(data_info[1:]):
        filename = x[0][1:-1] + ".wav"
        split = x[1]  # Train, Val, Test
        speaker = x[2]
        # gt = x[6:]
        gt = x[3:13]
        if "Test" in x[1]:
            continue
        for emo, score in zip(heads[-10:], gt):
            if float(score) == 1.0:
                type_ = emo
                break
        else:
            raise RuntimeError(f"{filename} has no emotion with score == 1.0, {gt}")
        split2files[split].append(np.hstack([filename, speaker, type_, gt]))

    # Encoder Speakers
    all_speakers = []
    for split in split2files.keys():  # 'Train', 'Val'
        for dt in split2files[split]:
            all_speakers.append(dt[1])

    speaker_encoder = LabelEncoder()
    all_encoded_speakers = speaker_encoder.fit_transform(all_speakers)
    encoded_speakers = {
        "Train": all_encoded_speakers[: len(split2files["Train"])],
        "Val": all_encoded_speakers[len(split2files["Train"]) :],
    }

    assert len(encoded_speakers["Train"]) == len(split2files["Train"])
    assert len(encoded_speakers["Val"]) == len(split2files["Val"])

    for split in split2files.keys():
        for dt, spk_id in zip(split2files[split], encoded_speakers[split]):
            dt[1] = f"{dt[1]}_{spk_id}"

    # Writing csv files
    for split, data in split2files.items():
        data.insert(0, np.array(heads))
        os.makedirs(save_path, exist_ok=True)
        np.savetxt(
            save_path / f"exvo_{split.lower()}.csv",
            np.array(data),
            delimiter=",",
            fmt="%s",
        )


#     # Run when speaker's id will become available on the test set
#     subject2files = defaultdict(list)
#     for i, x in enumerate(data_info[1:]):
#         split = x[1]
#         if 'Test' in split:
#             subject_id = x[2]
#             filename = x[0][1:-1] + '.wav'
#             gt = x[5:]
#             subject2files[subject_id].append(np.hstack([filename, gt]).tolist())

#     f = open('exvo_test_subject2files.json', 'w')
#     json.dump(subject2files, f)
#     f.close()


def create_test_splits(data_path, save_dir):
    df = pd.read_csv(data_path)
    df = df[df.Split == "Test"]
    emotion = df.columns[-10:]
    ft = df[~df.Awe.isna()]
    heads = [
        "id",
        "speaker",
        "type",
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]

    out = []
    for i, row in df.iterrows():
        id_ = row.File_ID[1:-1]
        speaker = row.Subject_ID
        item = [id_, speaker, np.nan] + [np.nan] * 10
        out.append(item)
    out = pd.DataFrame(out, columns=heads)
    print(f"test, {out.shape}")
    out_path = join(save_dir, "exvo_test.csv")
    out.to_csv(out_path, index=False)
    # ft
    out = []
    for i, row in ft.iterrows():
        id_ = row.File_ID[1:-1]
        speaker = row.Subject_ID
        type_ = None
        emo_buf = []
        for emo in emotion:
            value = row.loc[emo]
            if value == 1.0:
                type_ = emo
            emo_buf.append(value)
        assert type_ is not None, f"{id_} has no main emotion"
        item = [id_, speaker, type_] + emo_buf
        out.append(item)
    out = pd.DataFrame(out, columns=heads)
    print(f"ft, {out.shape}")
    out_path = join(save_dir, "exvo_ft.csv")
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.test:
        create_splits(Path(args.data_file_path), Path(args.save_path))
    else:
        create_test_splits(args.data_file_path, args.save_path)
