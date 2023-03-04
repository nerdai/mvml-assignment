## On the overall approach
- What has been adopted here is widely understood as the standard approach to
building classification models on text data: 
    - tokenize text (and filtering out stopwords), followed by some additional 
    processing to prepare them as inputs forthe classifation head
- The ML model used for this project was SVM with l2 regularization. A Naive's
Bayes Classifier was also tested, but the SVM was chosen due its slightly
superiour performance.
    - The overall performance of the model is fair, and likely can be improved
    upon. Though as mentioned in the assignment doc, this was activity was not
    de-prioritized in lieu of building an effective end-to-end pipeline.
- With respect to creation of the pipeline, the approach used here leveraged
the `Pipelines` class of `sklearn.preprocessing`.
    - These objects have a typical sklearn API and can be easily saved into a
    binary after training.
- In order to promote reproducibility and portability of the model and pipeline,
the package has been Dockerized.
- Standing up a service for the ML model was not considered here, though it's 
understood that this step can be made straightforward with the use of tools 
such as FastAPI, Kubernetes etc.

Note that the pipeline and source of inspiration for the ML modelling approach
is linked below: 
- <https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#working-with-text-data>

## On preprocessing steps

## On evaluation procedure and metrics

## On aspects that could be improved