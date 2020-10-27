from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from glm.glm import GLM
from glm.families import Gaussian
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(path):
    return pd.read_csv(path)

def get_cols(df,cols):
    return df[cols]

def inferential_regression(df):
    