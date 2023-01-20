"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

# Logistic Regression ========================== Behzad Amanpour ===================
from sklearn.linear_model import LogisticRegression 

model = LogisticRegression()
model.fit(X, y)
print(model.coef_, model.intercept_)
y_pred = model.predict( X )
prob = model.predict_proba(X)[:, 1]  # probability of "y=1" for each sample (row) in "X"
model.score( X, y) # accuracy
sum(y)

from sklearn.metrics import balanced_accuracy_score  # average of sensitivity & specificity 

balanced_accuracy_score(y,y_pred)
y_pred = model.predict_proba(X)[:, 1] > 0.35
balanced_accuracy_score(y,y_pred)

# Cross-Validation ============================= Behzad Amanpour ===================
from sklearn.model_selection import cross_val_score
import numpy as np

model = LogisticRegression()
scores = cross_val_score( model, X, y, scoring="balanced_accuracy", cv=3   )
print('balanced_accuracy:',np.mean(scores))

# Standardization & Normalization =============== Behzad Amanpour ===================
from scipy.stats import zscore
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import copy

X2 = copy.copy(X)    # I use "copy" because if you change "X", "X2" will not be chnaged
X = zscore( X)    # zscore(X, axis=1) calculates zscore in rows
scaler = MinMaxScaler()
X = scaler.fit(X).transform(X)
model = LogisticRegression()
scores = cross_val_score( model, X, y, scoring="balanced_accuracy", cv=3   )
print('balanced_accuracy:',np.mean(scores))
model = LogisticRegression()
model.fit(X, y)
coef = model.coef_
X = copy.copy(X2)

# PolynomialFeatures ============================ Behzad Amanpour ====================
# for more info, please visit "Linear Regression.pdf"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

degrees = [1,2,3,4,5,6,7,8,9]
for i in range(9):
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    model = LogisticRegression() # default, solver = 'lbfgs', max_iterint = 100
    # model = LogisticRegression(solver='liblinear')
    # model = LogisticRegression(solver='liblinear', max_iter=1000)
    pipeline = Pipeline(
        [   
            ("features", polynomial_features),
            ("model", model),
        ])
    print('degree: ',degrees[i])
    scores = cross_val_score( pipeline, X, y, scoring="balanced_accuracy", cv=3   )
    print('score:',np.mean(scores))

# Optimization ================================== Behzad Amanpour =====================
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline(
    [   
        ("features", PolynomialFeatures( include_bias=False)),
        ("model", LogisticRegression()),
    ])
param_grid = {
    'features__degree': [1, 2, 3, 4],
    'model__solver': ('liblinear', 'lbfgs'),
    'model__max_iter': [10, 100, 1000]
    }
gs = GridSearchCV(pipeline, param_grid, scoring='balanced_accuracy', cv=3)
gs.fit(X, y)
print(gs.best_params_)
print(gs.best_score_)

# Summary ======================================= Behzad Amanpour =====================
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score

model = LogisticRegression()
scores = cross_val_score( model, X, y, scoring="balanced_accuracy", cv=3   )
print('score:',np.mean(scores))

from sklearn.metrics import balanced_accuracy_score 

model.fit( X_train , y_train)
y_pred = model.predict( X_test )
balanced_accuracy_score( y_test , y_pred )
