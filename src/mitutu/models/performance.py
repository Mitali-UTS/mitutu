from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score
from typing import Optional

# function from class
def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

# function to calculate and display recall scores
def recall_scores(
    y_true,
    y_preds,
    target_labels: list,
    return_scores: bool = False) -> Optional[dict]:
    """
    Display recall scores per class and optionally return them.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_preds : array-like
        Estimated targets as returned by a classifier.
    target_labels : array-like
        List of class labels in the same order as used for recall calculation.
    return_scores : bool, default=False
        If True, return a dictionary of recall scores per class.

    Returns
    -------
    dict or None
        Dictionary of recall scores per class if return_scores=True, otherwise None.
    """
    print("RECALL SCORES")
    recall_per_class = recall_score(y_true, y_preds, labels=target_labels, average=None)

    for i, recall in enumerate(recall_per_class):
        print(f"Recall for class {target_labels[i]}: {recall:.4f}")

    if return_scores:
        return {label: score for label, score in zip(target_labels, recall_per_class)}

# function to calculate and display accuracy scores
def overall_accuracy(
    y_true, 
    y_preds, 
    return_score: bool = False) -> Optional[float]:
    """
    Display the overall accuracy score, and optionally return it.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_preds : array-like
        Estimated targets as returned by a classifier.
    return_score : bool, default=False
        If True, return the accuracy score as a float.

    Returns
    -------
    float or None
        The overall accuracy score if return_score=True, otherwise None.
    """
    score = accuracy_score(y_true, y_preds)

    print("ACCURACY SCORES")
    print(f"Overall Accuracy: {score:.4f}")

    if return_score:
        return score
    
# function to calculate and display precision scores
def overall_precision(
    y_true, 
    y_preds, 
    average: str = "macro",
    return_score: bool = False) -> Optional[float]:
    """
    Display the overall precision score, and optionally return it.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_preds : array-like
        Estimated targets as returned by a classifier.
    average : str, default="macro"
        Averaging method for precision. Options:
        - 'micro': global precision across all classes
        - 'macro': unweighted mean of per-class precision
        - 'weighted': weighted mean of per-class precision
        - 'binary': for binary classification
    return_score : bool, default=False
        If True, return the precision score as a float.

    Returns
    -------
    float or None
        The overall precision score if return_score=True, otherwise None.
    """
    score = precision_score(y_true, y_preds, average=average, zero_division=0)
    
    print("PRECISION SCORES")
    print(f"Overall Precision ({average}): {score:.4f}")

    if return_score:
        return score

# function to calculate and display f1 score
def f1_scores(
    y_true,
    y_preds,
    target_labels: list,
    return_scores: bool = False) -> Optional[dict]:
    """
    Display F1 scores per class and weighted F1 score, and optionally return them.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_preds : array-like
        Estimated targets as returned by a classifier.
    target_labels : array-like
        List of class labels in the same order as used for F1 calculation.
    return_scores : bool, default=False
        If True, return a dictionary containing per-class F1 scores and weighted F1 score.

    Returns
    -------
    dict or None
        Dictionary with per-class F1 scores and weighted F1 score if return_scores=True,
        otherwise None.
    """
    print("F1 SCORES")
    f1_per_class = f1_score(y_true, y_preds, labels=target_labels, average=None)

    scores_dict = {}
    for i, f1 in enumerate(f1_per_class):
        print(f"F1 score for class {target_labels[i]}: {f1:.4f}")
        scores_dict[target_labels[i]] = f1

    weighted_f1 = f1_score(y_true, y_preds, labels=target_labels, average="weighted")
    print(f"Weighted F1 score: {weighted_f1:.4f}\n")
    scores_dict["weighted"] = weighted_f1

    if return_scores:
        return scores_dict

# function to evaluate classififcation models
def evaluate_classification_model(y_true, 
                                  y_preds, 
                                  data_case, 
                                  target_labels = None, 
                                  target_names = None,
                                  display_accuracy = True,
                                  display_precision = True,
                                  display_recall = True,
                                  display_f1score = True,
                                  display_confusionmatrix = True):
    """Creates a classification report for the predictions which includes recall scores, f1-scores and confusion matrix.
    
    Parameters
    ----------
    y_true:
        The actual values of the dependent variable.
    y_preds:
        The predicted classification values. They should match up with the target labels.
    data_case: 
        This is printed to distinguish the report. Usually takes on values such as 'training', 'testing' or 'validation'
    target_labels: list
        This has to be a list of the actual numeric labels.
    target_names: list
        This has to be the label names, to be used for the confusion matrix corresponding to the target labels.
    
    Returns
    -------
    """

    # import useful libraries
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"Classification metrics: {data_case} \n")
    # order for labels encoded
    # we will be primarily interested in class 1 of data

    # display accuracy scores if true
    if display_accuracy:
      overall_accuracy(y_true, y_preds)
      print('\n')

    # display precision if true
    if display_precision:
      overall_precision(y_true, y_preds)
      print('\n')

    # display recall scores if true
    if display_recall:
      recall_scores(y_true, y_preds, target_labels)
      print('\n')
    
    # display f1 score if true
    if display_f1score:
      f1_scores(y_true, y_preds, target_labels)
      print('\n')
    
    # display confusion matrix if true
    if display_confusionmatrix:
      # create a figure with 1 row and 2 columns of subplots
      fig, axes = plt.subplots(1,2, figsize=(10,5))

      # confusion matrix
      cmap = "Greens"
      cm = confusion_matrix(y_true, y_preds, labels = target_labels)
      sns.heatmap(cm, annot=True, 
                  cmap =cmap,
                  xticklabels=target_names, 
                  yticklabels = target_names, 
                  fmt='g',
                  ax=axes[0])
      axes[0].set_xlabel('Predicted')
      axes[0].set_ylabel('Actual')
      axes[0].set_title('Confusion Matrix with Count')

      # normalized confusion matrix
      cm_normalized = np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
      sns.heatmap(cm_normalized, 
                  annot=True, 
                  cmap =cmap,
                  xticklabels=target_names, 
                  yticklabels = target_names,
                  vmin = 0, 
                  vmax = 1,
                  ax = axes[1])
      axes[1].set_xlabel('Predicted')
      axes[1].set_ylabel('Actual')
      axes[1].set_title('Normalized Confusion Matrix')

      # show plot
      plt.tight_layout()
      plt.show()