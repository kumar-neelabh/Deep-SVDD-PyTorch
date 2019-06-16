from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

class Classifier:
  '''
  An extension to the original Deep SVDD to enable classification of test cases.
  
  optim_thresh(): get the threshold at which your metric is maximized
  classifier(): get y_pred based on optimum threshold
  wrong_answers(): get the list of test examples the classifier() wrongly classified in order to optimize the metric
  '''
  
  def __init__(self, y_true, probs):
    self.y_true = y_true
    self.probs = probs
    self.y_pred = None
    self.acc_thresh = None
    self.max_acc = None
  
  def optim_thresh(self, score='acc'):
    if score == 'acc':
      fpr, tpr, thresholds = roc_curve(self.y_true, self.probs)
      accuracy_scores = []
      for thresh in thresholds:
          accuracy_scores.append(accuracy_score(self.y_true, 
                                               [1 if m > thresh else 0 for m in self.probs]))

      accuracies = np.array(accuracy_scores)
      max_acc = accuracies.max()
      max_acc_thresh =  thresholds[accuracies.argmax()]
      
      self.max_acc = max_acc
      self.acc_thresh = max_acc_thresh
      return max_acc_thresh

    return 'metric not defined'
  
  def classify(self, score='acc'):
    if score == 'acc':
      thresh = self.optim_thresh(score='acc')
      self.y_pred = [1 if m > thresh else 0 for m in self.probs]
      return self.y_pred

    return 'metric not defined'
  
  def wrong_answers(self, X, y):
    y, self.y_true = np.array(y), np.array(self.y_true)
    assert (self.y_true == y).all()
    
    wrong_idx = np.array([i for i, pred in enumerate(self.y_pred) if pred != self.y_true[i]]).flatten()
    wrong_imgs = [X[index] for index in wrong_idx]
    return wrong_imgs