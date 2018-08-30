# -*- coding: utf-8 -*-


#name:  MultiFactor.py
#author: XX
#create on: 20180810
# 20180824 主体框架完成. by xx
# 20180827 用pandas类型的数据改写（方便对其时间和代码）；添加注释.  by: xx


#import pandas as pd
#import numpy as np

#import scipy.stats as ss
#import statsmodels.api as sm

#import sklearn
#from sklearn import metrics
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix

#from sklearn.cross_validation import train_test_split


#import seaborn as sns
#import matplotlib.pyplot as plt

from MultiFactor import MultiFactor 

test = MultiFactor()
test.prepare_data()

test.factor_process()
test.return_process()
test.construct_panel()