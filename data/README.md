## About the Avazu Click Through Rate dataset

The dataset used for this project comes from the [Kaggle Avazu click through rate prediction competition](https://www.kaggle.com/c/avazu-ctr-prediction/data). We download the training set (see the **train.gz** file), unzip it, and find a ~6Gb file with web client interaction data, and a target variable called *click*.

### Subsetting `train.csv`

Because this dataset is so big (40 million rows!), we decided to pare it down to 5 million rows for speed. Here is bash code for how the dataset was formed.
```bash
sed 1d train.csv > train_no_head.csv
gshuf -n 5000000 train_no_head.csv > train_5mil_tmp.csv
head -n 1 train.csv > train_head.csv
cat train_head.csv train_5mil_tmp.csv > train_5mil.csv
rm train_no_head.csv train_head.csv train_5mil_tmp.csv
```

Instead of passing around a large data file between us though, we just save the `id` field (the identifier of each row) to a file. That way, you can just know the rows we used for this project. This is the `ids.csv` file in the `data` directory.

```bash
awk -F "\"*,\"*" '{print $1}' train_5mil.csv > ids.csv
```

