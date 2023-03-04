What follows below are some of my comments around the discussion topics that 
were prompted in the assignment doc. I acknowledge that this set is likely not
exhaustive, but I hope it represents a good starting point to identifying the
top opportunities to drive the system forward in a meaningful way.

## On the overall approach
- What has been adopted here is widely understood as the standard approach to
building classification models on text data: 
    - tokenize text (and filtering out stopwords), followed by some additional 
    processing to prepare them as inputs forthe classifation head
- The ML model used for this project was log-reg with elasticnet regularization. 
A Naive's Bayes Classifier, and an SVM were also tested, but the log-reg was 
chosen due its slightly superior performance.
    - The overall performance of the model is fair, and likely can be improved
    upon. Though as mentioned in the assignment doc, this was activity was not
    de-prioritized in lieu of building an effective end-to-end pipeline.
- With respect to creation of the pipeline, the approach used here leveraged
the `Pipelines` class of `sklearn.preprocessing`.
    - These objects have a typical sklearn API and can be easily saved into a
    binary after training.
- In order to promote reproducibility and portability of the model and pipeline,
the package has been Dockerized (and random state set to `SGDClassifier`).
- Standing up a service for the ML model was not considered here, though it's 
understood that this step can be made straightforward with the use of tools 
such as FastAPI, Kubernetes etc.

Note that the pipeline and source of inspiration for the ML modelling approach
is linked below: 
- <https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#working-with-text-data>

## On preprocessing steps
- As alluded to above, preprocessing steps were taken in order to transform
the raw text inputs into features on which a classifier can be trained (and
predicted).
- Specifically, we used `CountVectorizer` and `TfidfTransformer` from the
`sklearn.feature_extraction.text` module.
    - These perform bag-of-words tokenization and tf-idf normalization so as to
    not have too high of average counts problem with longer texts.
- In terms of cleaning the data, the following steps were taken:
    - removal of the entries with `text` as null
    - removal of entries with `text` as whitespace(s)
    - removal of entries with replicated pairs of `(text, label)`
- Also, only the `text` columns feature was used to train the ML model, which
was based upon the intuition that this feature was likely the one carrying the
most rich relationship to the `label` column (compared to `title` and `author`).
- Note that the approaches used here are meant to be justified for the sake of
building only an MVP model and the prioritization of the building a working,
complete pipeline.

## On evaluation procedure and metrics
- With respect to evaluating the model and data:
    - the removal of entries with null `text` entries only reduced the training
    set by 39 records—meaning the omission of this records is likely reasonable.
    - accuracy, f1-score, and precision were all considered, and it seemed that
    the dataset was quite balanced, and so, accuracy can be a reliable metric
    here.
        - Note: there is evidence to suggest overfitting since training metrics
        were significantly higher than that of the test metrics. So, beyond this
        assignment we could look to tune the model to make it more generalizable
        or robust to unseen observations.
    
## On aspects that could be improved
- In terms of modelling techniques:
    - Further investigations of the cleanliness of the data as well as typical
    ML model building activities like hyperparameter tuning, cross-validation
    can be used to build a more performant and robust ML model.
    - One could also leverage the other data in `title` and `author` by 
    tokenizing and some preprocessing followed by a concatenation procedure to
    prepare a single-vector input to the ML model.
    - Additionally, if more data became available, then one could use more
    elaborate and modern techniques such as those found in deep learning.
        - We could use for example tokenizers, embeddings, and trasnformers
        as feature preprocessing steps instead of the basic approaches adopted
        here.
        - The classification head could also be a neural network and we could 
        either do monolithic training where the weights of the embeddings and
        transformers be udpated during the training of the classification head.
    - Finally, and again with ample amounts of data, one could consider GraphML
    techniques to build a model that captures the potential dependence between
    training observations (i.e., violation of iid assumption)
        - Specifically, perhaps using a Graphical representation that captures
        the relation of `author` nodes between the `articles` would carry some
        relevant information to the downstream classification task.

- In terms of modelling pipeline development:
    - Automation and tracking of experiments (i.e., different parameter settings
    and modelling techniques) could be set up in an autoML kind of fashion.
        - Custom built tools can be written here or, if appropriate tools like
        MLFlow and Weights and Biases can be used.
    - We also probably want to log a few things:
        - the model weights
        - the training performances
        - etc. 
            - these can be stored in AWS S3 for example using AWS SDK
    - The Docker image is a bit clunky, and there are further optmizations that
    can be done there in order to slim those images down.
    - CI/CD practices can be employed here depending on the delivery mechanism
    and also the addition of appropriate Github Actions (or workflows).
    - Could probably benefit in adding more unit tests to this module as well. 
    - If more elaborate DL models were to be built, then one valid concern 
    would be on the latency when querying the model through a web service.
        - Here, gRPC is preferred over REST protocol.

- Others:
    - An important piece to any project involving ML/AI is that having to do
    with measurement and attribuition.
    - One of the principles that I believe is important to uphold is to ensure
    that what the system is getting measured on (i.e., success metric) is
    precisely the metric that the system is optimizing for. Specifically, 
    proxy metrics and other metrics should be avoided if possible, and if not
    should be handled with extreme care.