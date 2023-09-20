# Scaling Machine Learning Meodels Using Dask
This tutorial demonstrates how we can scale a machine learning model in Dask.

## Distributed Training

Distributed Training is most useful for training large models on medium-sized datasets.You may have a large model when searching over many hyper-parameters, or when using an [ensemble method](https://scikit-learn.org/stable/modules/ensemble.html) with many individual estimators.

To demonstrate distributed training we will be using the Newsgroup dataset available within [Scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html). We will be building an ML pipeline that will

1. [Tokenise the Text](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
2. [Normalize the data](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
3. [Implement an SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

Each of these pipeline step can have different hyperparameters that determines the accuracy of the model. It a good ide to do an xxhaustive search over a list of paremeters, in each step, to determine which hyper-parameter is best suited for a model. is the process of performing hyperparameter tuning in order to determine the optimal values for a given model. 

In this tutorial we use [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to determine which hyperparameters are best suited for the pipeline we have defined. Scoring hyperparameters using GridSearchCV involves a __“fit”__ and this is a very compute intensive operation. So we use distributed training using Dask for this.



