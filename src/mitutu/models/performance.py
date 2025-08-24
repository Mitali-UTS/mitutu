# Solution:
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

# function to evaluate classififcation models
def evaluate_classification_model(y_true, y_preds, data_case = 'training', target_labels = [0,1], target_names = []):
    """
    Creates a classification report for the predictions which includes recall scores, f1-scores and confusion matrix.
    
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
    from sklearn.metrics import confusion_matrix, recall_score, f1_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"Classification metrics for {data_case}: \n")
    # order for labels encoded
    # we will be primarily interested in class 1 of data

    print("RECALL SCORES")
    recall_per_class = recall_score(y_true, y_preds, labels = target_labels, average=None)
    for i, recall in enumerate(recall_per_class):
      print(f"Recall for class {target_labels[i]}: {recall}")
    print()

    # f1 score
    print("F1 SCORES")
    f1_per_class = f1_score(y_true, y_preds, labels = target_labels, average=None)
    for i, f1 in enumerate(f1_per_class):
      print(f"f1 score for class {target_labels[i]}: {f1}")
    print(f"f1 weighted score: {f1_score(y_true, y_preds, labels = target_labels, average='weighted')}")
    print()

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