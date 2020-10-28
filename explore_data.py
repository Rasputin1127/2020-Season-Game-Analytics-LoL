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

def inferential_regression(df, columns, dep_var):
    x = sm.tools.tools.add_constant(df[columns].values)
    y = df[dep_var]
    model = sm.OLS(y,x)
    return model.fit()

def check_for_homoscedasticity(results, columns, dep_var, alpha=0.005):
    x = sm.tools.tools.add_constant(df[columns].values)
    y_predict = results.predict(x)
    residuals = y_predict - y
    plt.scatter(y_predict, residuals, alpha)
    plt.show()
    return residuals

def qq_plot(residuals):
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.show()

def check_VIF_values(df, columns, results):
    x = df[columns]

    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns

    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))] 

    p_values = pd.DataFrame(results.pvalues)
    vif_data["p_values"] = p_values.values[1:]

    vif_data.head()
    return vif_data

def plot_beta(alpha, beta, ax, title=None, label=None, xticks=[0.0, 0.5, 1.0]):

    # Build a beta distribtuion scipy object.
    dist = stats.beta(alpha, beta)

    # The support (always this for the beta dist).
    x = np.linspace(0.0, 1.0, 10001)

    # The probability density at each sample support value.
    y = dist.pdf(x)

    # Plot it all.
    lines = ax.plot(x, y, label=label)
    ax.fill_between(x, y, alpha=0.2, color=lines[0].get_c())
    if title:
        ax.set_title(title)
    ax.get_yaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([np.max(y)])
    ax.get_xaxis().set_ticks(xticks)
    ax.set_ylim(0.0, np.max(y)*1.2)

def get_beta_dist_params(dist_samples):
    dist_samples = np.array(dist_samples)
    num_conversions = dist_samples.sum()
    total_visitors = len(dist_samples)
    alpha = num_conversions + 1
    beta = (total_visitors - num_conversions) + 1
    mean = 1 * num_conversions / total_visitors
    return alpha, beta, mean, num_conversions, total_visitors

def plot_beta_dist(website_samples, ax, label=None, xlim=(0,1)):
    alpha, beta, mean, num_conversions, total_visitors = get_beta_dist_params(website_samples)
    title = None if label else r"Converted {}/{}".format(num_conversions, total_visitors)
    plot_beta(alpha, beta, ax, title, label, [0.0, mean, 1.0])
    ax.set_xlabel("Win Rate")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(xlim)

def get_samples(df, columns, param1, param2, scale=0):
    new_df = df[columns]
    a = new_df[new_df[param1] > new_df[param2] + scale]
    return a[columns[0]]




if __name__=='__main__':
    chall_df = get_data('Data/Challenger_Ranked_Games.csv')
    gm_df = get_data('Data/GrandMaster_Ranked_Games.csv')
    m_df = get_data('Data/Master_Ranked_Games.csv')

    chall_df_clean = chall_df[chall_df['gameDuraton'] > 600]
    gm_df_clean = gm_df[gm_df['gameDuraton'] > 600]
    m_df_clean = m_df[m_df['gameDuraton'] > 600]

    columns = ['blueWins','blueWardPlaced', 'redWardPlaced']
    
    for i in range(0,26):
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        a_vision_samples = get_samples(chall_df_clean, columns, 'blueWardPlaced', 'redWardPlaced',i)
        plot_beta_dist(a_vision_samples, ax, label=f"Vision Greater by {i}",xlim=(0.6,0.7))
        ax.legend()
        plt.savefig(f'images/chall_{i}.png')
        plt.show()