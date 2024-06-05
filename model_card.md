# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model used is a RandomForestClassifier, which is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


## Intended Use
This model is intended to be used on the census dataset.

## Training Data
The training data used the category features: "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"

## Evaluation Data
20% of the original dataset was randomly split for use in training and evaluation, using: train_test_split(data, test_size=0.20, random_state=20). The evaluation data is stored in the file: slice_output.txt

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7255 | Recall: 0.6321 | F1: 0.6756

## Ethical Considerations
This model uses a census dataset, and may contain biases.

## Caveats and Recommendations
The model should continue to be trained and tested to ensure accurancy and reduce any biases.
