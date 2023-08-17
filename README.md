  ## Install Packages

```bash
pip3 install flair
```
```bash
pip3 install seqeval
```
```bash
pip3 install sentence-transformers
```

## Dataset
You can download the dataset from the ``link"

## Partitioned Data Creation
Run the following script to generate the partition data.
```bash
python3 partition.py
```

## Training
Run the following script to train the model.

```bash
python3 train.py --dataset_path data\
--data_train train.txt\
--data_test test.txt\
--data_dev valid.txt\
--output_dir model\
--model_name_or_path allenai/scibert_scivocab_cased\
--layers -1\
--subtoken_pooling first_last\
--hidden_size 256\
--learning_rate 5e-05\
--use_crf True
```

## Inferencing
Run the following script to test the best model.

```bash
python3 test.py --dataset_path data \
--data_train train.txt\
--data_test test.txt\
--data_dev valid.txt\
--load_trainedm_dir model/best-model.pt\
--pred_txt_fl prediction.txt\
--label_dict dict_nw.pkl\
--result_file result.txt\
--model_name_or_path allenai/scibert_scivocab_cased
