
1. Pre-processing the data
python3 preprocess.py --src_dir /path/to/wav --tgt_dir /path/to/output/dir

2. Splitting the data into train and valid subsets
python3 create_splits.py --data_file_path=../data/labels/data_info.csv --save_path=../data/labels/filelists

3. Training
python3 train.py




