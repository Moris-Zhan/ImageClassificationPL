
def accuracy_score(self, y_true, y_pred):
    return (y_true == y_pred).sum() / float(len(y_true))    