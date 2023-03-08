#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> Homework 9 Programming Task (10 Points) </h1>
# <h3 align="center"> IST 5520, Fall 2022 </h3>
# <h3 align="center"> By: Estelle Lu </h3>
# <h3 align="center"> Last Update: 10/30/2022 </h3>

# The data file *Banks.csv* contains data on a sample of 20 banks. The “Financial Condition” column records the judgment of an expert on the financial condition of each bank. This outcome variable takes one of two possible values — 1 (weak) or 0 (strong) — according to the financial condition of the bank. The predictors are two ratios used in the financial analysis of banks: 
# 
# - **TotLns&Lses/Assets** is the ratio of total loans and leases to total assets;
# - **TotExp/Assets** is the ratio of total expenses to total assets. The target is to use the two ratios for classifying the financial condition of a new bank.

# ## Step 1: Model Specification (3 points)
# 
# Use LaTeX syntax (for an introduction, refer to https://patrickwalls.github.io/mathematicalpython/jupyter/latex/) to write the estimated equation that associates the financial condition of a bank with its two predictors in three formats:
# 
# 1)	The logit as a function of the predictors (1 point)
# 
# Your answer: (See bellow)
# 
# 2)	The odds as a function of the predictors (1 point)
# 
# Your answer: (See bellow)
# 
# 3)	The probability as a function of the predictors (1 point)
# 
# Your answer: (See bellow)

# %1) The logit as a function of the predictors:
# \begin{equation}
#   \text{logit}(p)=\ln{(\frac{p}{1-p})}={\beta_{0}+\beta_{1} \text{x}_1 \text{+ ...} +\beta_{k} \text{x}_k} 
# \end{equation}

# %2) The odds as a function of the predictors:
# \begin{equation}
#   \frac{p}{1 - p}=e^{-(\beta_{0}+\beta_{1} \text{x}_1 \text{+ ...} +\beta_{k} \text{x}_k)} 
# \end{equation}

# %3) The probability as a function of the predictors:
# \begin{equation}
#  \text{p}=\frac{1}{1+e^{-(\beta_{0}+\beta_{1} \text{x}_1 \text{+ ...} +\beta_{k} \text{x}_k)} }
# \end{equation}

# ## Step 2: Logistic Regression (5 points)
# 
# Implement a logistic regression model (on the entire dataset) that models the status of a bank as a function of the two financial measures provided. Interpret the estimates of coefficients.
# 
# Note: Feel free to add more code lines below to implement the logic.

# In[2]:


# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# reading data from the csv
dat = pd.read_csv('/Users/estellelu/downloads/banks.txt')
dat.head()


# In[3]:


# Descriptive statistics
dat.describe().transpose()


# Calculate marginal effects and interpret the estimates of marginal effects.

# In[5]:


dat['Financial Condition'].value_counts()


# In[6]:


sns.countplot(x="Financial Condition", data=dat)


# In[7]:


# Above means the data of "Finacial Condition" is balanced
# Drop TotCap/Assets from the dataset, which is not used in the process
X = dat
X = X.drop(['TotCap/Assets','Financial Condition'], axis = 1)
X.head()
#X


# In[8]:


y = dat['Financial Condition']
y.head()


# In[9]:


from sklearn.model_selection import train_test_split

# 20-80% simple split
# To make the result reproducible, set the random_state
train_y,test_y,train_X,test_X = train_test_split(y, X,
                                                 test_size=0.2,
                                                 stratify=y)


# In[10]:


#train_y
train_y.shape


# In[11]:


test_y
#test_y.shape


# In[12]:


#train_X
train_X.shape


# In[13]:


test_X.shape


# In[14]:


train_y.describe()


# In[15]:


test_y.describe()


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


logit1 = LogisticRegression(solver='liblinear')
logit1.fit(train_X, train_y)


# In[18]:


pred_y_logit1 = logit1.predict(test_X)


# In[19]:


# Define a function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.plot()


# In[20]:


from sklearn import metrics

# Compute confusion matrix
cnf_matrix_logit1 = metrics.confusion_matrix(pred_y_logit1, test_y, labels=[1,0])

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix_logit1, classes=['Weak','Strong'],
                      title='Confusion matrix, without normalization')


# In[21]:


# Print classification report
print(metrics.classification_report(test_y,pred_y_logit1))


# We can see that only 50% customers who actually accepted "Financial Condition" of each bank whether "week" or "strong" have been correctly classified by the logistic regression model. In practice, if it is imperative to predict customers who would like to accept "Financial Condition", but were not correctly classified by the predictive model.
# 
# The “Financial Condition” column records the judgment of an expert on the financial condition of each bank.  The predictors are two ratios used in the financial analysis of banks: 
# - **TotLns&Lses/Assets** is the ratio of total loans and leases to total assets;
# - **TotExp/Assets** is the ratio of total expenses to total assets. 
# The two ratios to be used here for classifying the financial condition of a new bank.

# ## Step 3. Predict New Data (2 points)
# 
# Consider a new bank whose total loans and leases/assets ratio = 0.6 and total expenses/assets ratio = 0.11. From your logistic regression model, estimate the following two quantities for this bank: 
# 
# - the probability of being financially weak;
# - the classification of the bank (use cutoff = 0.5).
# 

# In[22]:


import statsmodels.api as sm
#logit_model=sm.Logit(y,X)
logreg = LogisticRegression()
logreg.fit(X, y)
log_odds = logreg.coef_[0]
print("Intercept: ",logreg.intercept_)
pd.DataFrame(log_odds, 
             X.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
#print("Intercept: ",logreg.intercept_)


# 
# %3) The probability of being financially weak is:
# % (leases/assets ratio = 0.6 and total expenses/assets ratio = 0.11)
# \begin{equation}
#  \text{p}=\frac{1}{1+e^{-(\beta_{0}+\beta_{1} \text{x}_1 +\beta_{2} \text{x}_2)} }
# \end{equation}
# \begin{equation}
#  \text{p}=\frac{1}{1+e^{-(12.50853293 + 0.039397 * 0.6 + 0.037424*0.11)} }
# \end{equation}

# In[23]:


import math
#plug in the intercept, coefficients, and financial ratios
pw = -(12.50853293+0.039397*.6+0.037424*.11)
p=1/(1+ math.exp(pw))
print ("The probablity = ", p)


# In[ ]:


# 1. The probability of being financially week is 99.99%, means this new banck's financial condition is weak.


# In[ ]:


# The classification of the bank


# In[24]:


# Print classification report
print(metrics.classification_report(test_y,pred_y_logit1))

