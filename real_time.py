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

