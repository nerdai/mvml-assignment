# Overview
Listed below are the high-level descriptions taken to build the ML model.

1. Training data was filtered to remove any observations with empty `text` field
2. The text field was tokenized using Bag of Words and after TD-IDF
3. An SVM model was built on top of the tokenized features.

The Python library used here was `sklearn`. In particular we made use of the
`Pipeline` constructs in order to stack the two transformers to perform the
tokenization of step 2 as well as the final predictive SVM model.

## The Project Structure
```
mvml-assignment/
    ¹²data/
        labels.csv
        test.csv
        train.csv
        clean_train.csv
    src/
        __init__.py
        data_cleaner.py
        get_metrics.py
        inference.py
        training.py
    tests/
        __init__.py
    utils/
        data_loader.py
        __init__.py
    notebooks/
        sandbox.ipynb
    setup.sh
    poetry.lock
    pyproject.toml
    Dockerfile
    ¹artifacts/
        models/
            model.joblib
        preds/
            predictions.csv
```
¹ `data` and `artifacts` are not checked into Github

² `data` directory and its data files need to be placed within this structure
to be able to run the pipeline.

#### Summmary of Code
- `data_cleaner.py` is used to filter out the bad training observations from the
`train.df` and store the resulting DataFrame as a `clean_train.csv`.
- `training.py` fits the sklearn `Pipeline` on the `clean_train.csv` and dumps
the binary model in `artifacts/models` directory.
- `inference.py` loads the trained model and takes as input a new data file
to make batch predictions that are stored in local in the `artifacts/preds`
directory.
- `get_metrics.py` takes a `predictions.csv` file and `labels.csv` and produces
the classification report.
- `Dockerfile` used dockerize the ML model for ease of deployment and
reproducibility.

#### Model Performance
The current model (and pipeline) achieves the below metrics on the test set:

```
              precision    recall  f1-score   support

           0       0.60      0.66      0.63      2339
           1       0.70      0.63      0.66      2854

    accuracy                           0.65      5193
   macro avg       0.65      0.65      0.65      5193
weighted avg       0.65      0.65      0.65      5193
```

## Commands
### Without Docker

As a an initial step, run `setup.sh` which creates the required directories.

Cleaning data:
```
poetry run python -m src.data_cleaner
```

Training model:
```
poetry run python -m src.training 
```

Inference with model:
```
poetry run python -m src.inference --data ./data/fake_news/test.csv
```

Get metrics of test set:
```
poetry run python -m src.get_metrics --labels ./data/fake_news/labels.csv \
--preds ./artifacts/preds/predictions.csv
```


### With Docker

**Build docker image**
```
docker build -f Dockerfile . -t mvml-assignment
```

**Run docker image (inference)**
```
# inference
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/preds/":/artifacts/preds \
mvml-assignment poetry run python -m src.inference \
--data ./data/fake_news/test.csv

# get metrics
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/preds/":/artifacts/preds \
mvml-assignment poetry run python -m src.get_metrics \
--labels ./data/fake_news/labels.csv --preds ./artifacts/preds/predictions.csv 
```
