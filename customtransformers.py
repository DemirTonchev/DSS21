"""
importing libraries 
importing libary is done by using:
  - "import libname" or "import libname as lib"
then you can use lib.func to access the specific function you want. 
if you want to import just one function:
    - "from libname import func"
    - DO NOT USE "from libname import *" as this might have unwanted consequences

"""
# good 
import pandas as pd
# bad:
# from pandas import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import FunctionTransformer


#%%
# =============================================================================
# read data from csv and basic pandas commands
# =============================================================================

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
                   delim_whitespace=True,
                  names = ['mpg',          # continuous
                          'cylinders',     # multi-valued discrete
                          'displacement',  # continuous
                          'horsepower',    # continuous
                          'weight',        # continuous
                          'acceleration',  # continuous
                          'model_year',    # multi-valued discrete
                          'origin',        # multi-valued discrete
                          'name',          # string (unique for each instance))
                          ],
                  na_values = '?',
                  )

#%%
# you can select one column such as 
data['mpg']
# or 
data.loc[:, 'mpg']
# several columns as
data[['mpg','horsepower']]

data.loc[:, ['mpg','horsepower']]

# select using boolean mask 
mask = np.random.choice([True, False], size=len(data), p=[0.05, 0.95])
data.loc[mask, ['mpg','horsepower']]

mask = data['mpg'] > 20
data.loc[mask, ['mpg','horsepower']].mean()
# or more useful
mask = data.isna().any(axis=1)
data.loc[mask, :]

#%%
# =============================================================================
# Vectorization in python (do not write for loops)
# https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html
# =============================================================================

array = np.random.randint(1, 10, size = 10000)
#%%
# %%timeit -n 100 avg = 0
# for i in array:
#     avg += i
# avg = avg/len(array)

#%%
# %timeit -n 100 np.mean(array)
    
#%%
dum_data = pd.DataFrame([[-1, 2, 'a'], [-0.5, 6, 'a'], [0, 10, 'b'], [1, 18, 'b']], columns = ['a','b','c'])
y = np.array([1,2,3,4])

#%%
# =============================================================================
# data preparation/feature engineering 
# =============================================================================
# lets standartize the data https://en.wikipedia.org/wiki/Feature_scaling

X = dum_data[['a','b']].copy()
X = X - X.mean()
X = X/X.std(ddof=0)

#%% 
## or do this in a function 

def standartize(X):
    return (X - X.mean() )/ X.std()
X = standartize(dum_data[['a','b']].copy())
print(X)

#%%
# basic feature engineering  with sklearn

X = dum_data[['a','b']].copy()

skscaler = StandardScaler()

skscaler.fit(X)
skscaler.transform(X)

skscaler.fit_transform(X)

#%%
from sklearn.preprocessing import PolynomialFeatures

X = dum_data[['a','b']].copy()
poly = PolynomialFeatures(include_bias = False)
poly.fit_transform(X)

#%%
# =============================================================================
# use custom function
# =============================================================================
from sklearn.preprocessing import FunctionTransformer

# funtran = FunctionTransformer(func = np.exp, inverse_func = np.exp)
def id_fun(X):
    return X
funtran = FunctionTransformer(func = id_fun)

funtran.fit(X)
funtran.transform(X)

funtran.inverse_transform( funtran.transform(X) )

#%% Lets create transformers ourselves
# or what to do if we need something that is not in sklearn
from sklearn.utils import check_array
from sklearn.base import TransformerMixin, BaseEstimator

# classes and interfaces for more info:
# https://scikit-learn.org/stable/developers/develop.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# Custom transformers

def checkNA(method):
    def wrapper(self, X):
        if np.isnan(X).any():
            raise Exception("There are missing values in the data")
        else:
            method(self, X)
    return wrapper

class CustomStandardScaler(TransformerMixin, BaseEstimator):
    
    @checkNA
    def fit(self, X, y=None):
        # X = self._validate_data(X, estimator = self)
        self.means = np.mean(X, axis = 0)
        self.vars  = np.var(X, axis = 0)
        self.scale = np.sqrt(self.vars)
        
        return self
    
    def transform(self, X):
        # X = self._validate_data(X, estimator = self)
        X = X - self.means
        X = X/self.scale
        return X


scaler = CustomStandardScaler()
scaler.fit(X.values)

#%%
skscaler.fit_transform(X)
#%%
# =============================================================================
# crate custom minmaxscaler, custom mean imputer.. custom whatever you need.
# list of available preprocesing
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# =============================================================================


class ColSelector(TransformerMixin, BaseEstimator):
    """selects columns from pandas dataframe
    """
    def __init__(self, columns):
        pass    
        
    def fit():
        pass
    
    def transform():
        pass

class CustomTranformer(TransformerMixin, BaseEstimator):
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X):
        pass


#%%
# =============================================================================
# Sklearn transfomers as part of pipeline
# =============================================================================

# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyRegressor, DummyClassifier

poly = PolynomialFeatures(include_bias = False)
skscaler = StandardScaler()

X_poly = poly.fit_transform(X)
X_scaled = skscaler.fit_transform(X_poly)

#%%
pipe = Pipeline([ ('poly', PolynomialFeatures(include_bias = False) ) ,
                    ('scaler', StandardScaler() ),
                   # ('custom', CustomTranformer() )
                    ('model', DummyRegressor()),
                ]
                )


# pipe.fit_transform(X)

pipe.fit(X, y)
pipe.predict(X)

#%% 
# Onehot encoder transformer 
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse = False, handle_unknown='ignore')
onehot.fit_transform(dum_data[['c']])

#%%
# Feature union use case 

baby_pipe = FeatureUnion([
                        ('numerical',
                        Pipeline([
                            ('select_num', FunctionTransformer(func = lambda X: X.loc[:, ['a','b']])),
                            ('poly',       PolynomialFeatures(include_bias = False)),
                            ('scaler',     StandardScaler()),
                                ])
                        ),
                         ('categorical', 
                          Pipeline([
                              ('pass_cat', FunctionTransformer(func = lambda X: X.loc[:, ['c']]) ),
                              ('onehot',   OneHotEncoder(sparse = False, handle_unknown='ignore') )
                        ] ) 
                         ),
                    ])
                        
baby_pipe.fit_transform(dum_data)

#%%
from sklearn.linear_model import LinearRegression

super_pipe = Pipeline([ ('baby_pipe', baby_pipe),
                        # ('model', LinearRegression() ),
                        ('model', DummyRegressor() )
                        ])

super_pipe.fit(dum_data, y)
super_pipe.predict(dum_data)

# =============================================================================
# Big task 
# create train and test pipeline for your project
# create benchmark - w dummyregressor
# create some other model and 
# =============================================================================
#%%

from sklearn.model_selection import train_test_split

target = data['weight']

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

num_cols = ['mpg', 'displacement']
cat_cols = ['cylinders']

baby_pipe = FeatureUnion([
                        ('numerical',
                        Pipeline([
                            ('select_num', FunctionTransformer(func = lambda X: X.loc[:, num_cols])),
                            ('poly',       PolynomialFeatures(include_bias = False)),
                            ('scaler',     StandardScaler()),
                                ])
                        ),
                         ('categorical', 
                          Pipeline([
                              ('pass_cat', FunctionTransformer(func = lambda X: X.loc[:, cat_cols]) ),
                              ('onehot',   OneHotEncoder(sparse = False, handle_unknown='ignore') )
                        ] ) 
                         ),
                    ])
                        
baby_pipe.fit_transform(X_train)

#%%
from sklearn.linear_model import LinearRegression

super_pipe = Pipeline([ ('baby_pipe', baby_pipe),
                        ('model', LinearRegression() ),
                        # ('model', DummyRegressor() )
                        ])

super_pipe.fit(X_train, y_train)

super_pipe.predict(X_test)



