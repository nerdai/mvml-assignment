# Overview
Listed below are the high-level descriptions taken to build the ML model.

1. Training data was filtered to remove any observations with empty `text` field
2. The text field was tokenized using Bag of Words and subsequently TF-IDF
3. A `log_reg` model was built on top of the tokenized features.

The Python library used here was `sklearn`. In particular we made use of the
`Pipeline` constructs in order to stack the two transformers to perform the
tokenization of step 2 as well as the final predictive log-reg model.

![image](https://user-images.githubusercontent.com/92402603/222918732-74131524-5c16-4cd4-aa32-333befe4b860.png)

## The Project Structure
```
mvml-assignment
├── setup.sh
├── artifacts¹
│   ├── models
│   │   └── model.joblib
│   └── preds
│       └── predictions.csv
├── data¹²
│   └── fake_news
│       ├── labels.csv
│       ├── test.csv
│       ├── train.csv
│       └── clean_train.csv
├── COMMENTARY.md
├── Dockerfile
├── README.md
├── .gitignore
├── poetry.lock
├── pyproject.toml
├── notebooks
│   └── sandbox.ipynb
├── utils
│   ├── data_loader.py
│   ├── __init__.py
├── tests
│   ├── __init__.py
│   └── test_mvml_assignment.py
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── training.cpython-311.pyc
    │   ├── inference.cpython-311.pyc
    │   ├── get_metrics.cpython-311.pyc
    │   ├── __init__.cpython-311.pyc
    │   └── data_cleaner.cpython-311.pyc
    ├── data_cleaner.py
    ├── get_metrics.py
    ├── inference.py
    └── training.py
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
- `test_mvml_assignment.py` contains some unit test for the package and its
modules.
- `Dockerfile` for dockerize'ing the ML model for ease of deployment and
reproducibility.
- `COMMENTARY.md` contains discussions around the requested topics (i.e., on
overall approach, preprocessing, performance and metrics, and aspects that
can be improved).

#### Model Performance
The current model (and pipeline) achieves the below metrics on the
training and test set, respectively:

```
-----------------------TRAINING----------------------
              precision    recall  f1-score   support

           0       0.83      0.82      0.83     10387
           1       0.82      0.82      0.82     10218

    accuracy                           0.82     20605
   macro avg       0.82      0.82      0.82     20605
weighted avg       0.82      0.82      0.82     20605
```

```
-----------------------TEST--------------------------
              precision    recall  f1-score   support

           0       0.65      0.74      0.69      2339
           1       0.76      0.67      0.71      2854

    accuracy                           0.70      5193
   macro avg       0.70      0.71      0.70      5193
weighted avg       0.71      0.70      0.70      5193
```

## Commands
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

### Without Docker

Note that `poetry` (`v. ^1.3.0`) cli must be installed and the active
python version must be `v. ^3.11.0` if running without docker. 

Run setup:
```
sh setup.sh
```

Install (or upgrading) `poetry`:
```
# for linux, macOS
curl -sSL https://install.python-poetry.org | python3 -

# windows (powershell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Install dependencies:
```
poetry install
```

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

### Testing
```
poetry run pytest ./tests/*
```
