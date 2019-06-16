import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

class Classifier:
  '''
  An extension to the original Deep SVDD to enable classification of test cases.
  
  optim_thresh(): get the threshold at which your metric is maximized
  classifier(): get y_pred based on optimum threshold
  wrong_answers(): get the list of test examples the classifier() wrongly classified in order to optimize the metric
  '''
  
  def __init__(self, X, y_true, probs):
    self.y_true = y_true
    self.probs = probs
    self.y_pred = None
    self.acc_thresh = None
    self.max_acc = None
    self.X = X
    self.confusion_matrix = None
  
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
  
  def wrong_answers(self):
    if self.y_pred is None:
      _ = self.classify()
      
    wrong_idx = np.array([i for i, pred in enumerate(self.y_pred) if pred != self.y_true[i]]).flatten()
    wrong_imgs = [self.X[index] for index in wrong_idx]
    
    return wrong_imgs
  
  def confusion_matrix(self):
    if self.y_pred is None:
      _ = self.classify()
      
    conf_matrix = confusion_matrix(self.y_true, self.y_pred)
    return conf_matrix
 
