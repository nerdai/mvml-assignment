### Commands
Training model:
```
poetry run python -m src.training 
```

Inference with model:
```
poetry run python -m src.inference
```

### With Docker

**Build docker image**
```
docker build -f Dockerfile . -t mvml-assignment
```

**Run docker image (inference)**
```
# inference
docker run -v <local-data-path>:/data -v <local-preds-path>:/preds mvml-assignment poetry run python -m src.inference --data ./data/fake_news/test.csv

# get metrics
docker run -v <local-data-path>:/data -v <local-preds-path>:/preds mvml-assignment poetry run python -m src.get_metrics --labels ./data/fake_news/labels.csv --preds ./preds/preditiction.csv
```


