# Scaling Machine Learning Meodels Using Dask
This tutorial demonstrates how we can scale a machine learning model in Dask.

## Distributed Training

Distributed Training is particularly beneficial for training large models with medium-sized datasets. This scenario becomes relevant when dealing with extensive hyperparameter exploration or employing [ensemble method](https://scikit-learn.org/stable/modules/ensemble.html) involving numerous individual estimators.

To illustrate the concept of distributed training, we will utilize the [Newsgroup dataset]((https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)) from Scikit-learn. Our objective is to construct a machine learning pipeline that performs the following tasks:

1. [Tokenise the Text](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
2. [Normalize the data](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
3. [Implement an SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

Each of these pipeline steps can possess distinct hyperparameters that significantly influence the model's accuracy. It is highly advisable to conduct a comprehensive search across a range of parameters within each step to identify the most suitable hyperparameters for the model. This process, known as hyperparameter tuning, is essential for optimizing the model's performance.

In this tutorial, we leverage [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to identify the most appropriate hyperparameters for the defined pipeline. Assessing hyperparameters with GridSearchCV involves a "fit" operation that demands substantial computational resources. Therefore, we employ distributed training using Dask to efficiently distribute the computational workload.



