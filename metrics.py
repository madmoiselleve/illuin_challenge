import numpy as np
import pandas as pd

def precision(y_pred: np.array, y_true: pd.Series) -> float:
    ''' Input : vector of true context ids, vector of context id predictions
        Output : precision = number of right predictions / Total predictions
    '''
    if len(y_pred) > 0:
        return sum(y_pred.squeeze() == y_true.values)/len(y_true)
    else: return 0

def top_accuracy(y_true, y_pred) -> float:  ## Utile si on prédit plusieurs contexts pr 1 question
    right, count = 0, 0
    for y_t in y_true:
        count += 1
        if y_t in y_pred:
            right += 1
    return right / count if count > 0 else 0

def balanced_prec(y_pred: np.array, y_true: pd.Series) -> float:
    ''' Input : vector of true context ids, vector of context id predictions
        Output : balanced precision = sum( number of right predictions context c / nb of appearances of context c) / Total predictions
    '''
    balanced_correct = y_true[y_true == y_pred.squeeze()].value_counts().div(y_true.value_counts(), fill_value = 0)

    return sum(balanced_correct)/len(y_true)
