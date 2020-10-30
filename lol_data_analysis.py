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
    # ax.get_yaxis().set_ticks([np.mean(y)])
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

def plot_beta_dist(website_samples, ax, label=None, xlim=(0,1),lst=None, xtick=(0,1)):
    alpha, beta, mean, num_conversions, total_visitors = get_beta_dist_params(website_samples)
    title = None if label else r"Converted {}/{}".format(num_conversions, total_visitors)
    plot_beta(alpha, beta, ax, title, label, [xtick[0], mean, xtick[1]])
    if lst:
        lst.append(mean)
    ax.set_xlabel("Win Rate")
    ax.set_ylabel("Probability Density")
    # ax.get_yaxis().set_ticks([0,100])
    ax.set_xlim(xlim)

def get_samples(df, columns, param1, param2, sample_col=None, scale=0):
    new_df = df[columns]
    a = new_df[new_df[param1] > new_df[param2] + scale]
    if sample_col:
        samples = a[sample_col]
        return samples
    else:
        return a[columns[0]]

def gold_graph(df, columns, param1, param2, sample_col, title="Gold Graph"):
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    tick_list = [0]
    for i in range(0,3001,1000):
        chall_samples = get_samples(df, columns, param1, param2, sample_col,i)
        plot_beta_dist(chall_samples, ax, label=f"Gold Greater by {i}", lst=tick_list)
    tick_list.append(1)
    ax.get_xaxis().set_ticks(tick_list)
    ax.set_xlim(0.97,1)
    ax.tick_params(axis='x', rotation=65)
    ax.legend()
    plt.title(title)
    # plt.savefig(f"images/{title}.png")
    plt.show()

def get_linear_reg(df,columns,dep_var):
    x = sm.tools.tools.add_constant(df[columns].values)
    y = df[dep_var]
    model = sm.OLS(y,x)
    results = model.fit()
    print(results.summary())
    return results,x,y

def plot_linear_reg(results,x,y,title="Test for Homoscedasticity",filename1="images/homo_graph.png",filename2="images/qq_plot.png"):
    y_predict = results.predict(x)
    residuals = y_predict - y
    fig,ax = plt.subplots(1,1)
    ax.scatter(y_predict,residuals, alpha=0.01)
    plt.title(title)
    ax.set_xlabel("Gold")
    ax.set_ylabel("Variance")
    ax.set_ylim(-15000,15000)
    ax.set_xlim(15000, 80000)
    plt.tight_layout()
    plt.savefig(filename1)
    plt.show()

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.tight_layout()
    plt.savefig(filename2)
    plt.show()

def make_gif_of_graphs(df, columns, param1, param2, iterations, title="Vision Graph", filename=None,scal=0):
        for i in range(0,iterations+1):
            fig, ax = plt.subplots(1,1,figsize=(4,4))
            if scal:
                a_vision_samples = get_samples(df,columns, param1, param2, scale=scal*i)
            else:
                a_vision_samples = get_samples(df, columns, param1, param2,scale=i)
            plot_beta_dist(a_vision_samples, ax, label=f"Vision Greater by {i}",xlim=[0.6,0.7],xtick=(0.6,0.7))
            ax.legend()
            ax.set_title(title)
            ax.tick_params(axis='x',rotation=65)
            if not filename:
                pass
                # plt.savefig(f'images/graph_{i}.png')
            else:
                pass
                # plt.savefig(f'images/{filename}_{i}.png')

if __name__=='__main__':

    # Reading data into pandas DataFrames
    chall_df = get_data('Data/Challenger_Ranked_Games.csv')
    gm_df = get_data('Data/GrandMaster_Ranked_Games.csv')
    m_df = get_data('Data/Master_Ranked_Games.csv')

    # Cleaning the data to exclude games that are too short and produce outlier data
    chall_df_clean = chall_df[(chall_df['gameDuraton'] > 600) & (chall_df['gameDuraton'] < 1800)].copy()
    gm_df_clean = gm_df[(gm_df['gameDuraton'] > 600) & (gm_df['gameDuraton'] < 1800)].copy()
    m_df_clean = m_df[(m_df['gameDuraton'] > 600) & (m_df['gameDuraton'] < 1800)].copy()

    # Getting a gold graph for each tier of play that shows how well gold correlates with victory
    columns = ['blueWins','blueTotalGold','redTotalGold']
    gold_graph(chall_df_clean, columns, 'blueTotalGold', 'redTotalGold', 'blueWins', "Challenger Gold Graph")
    gold_graph(gm_df_clean, columns, 'blueTotalGold', 'redTotalGold', 'blueWins', "Grand Master Gold Graph")
    gold_graph(m_df_clean, columns, 'blueTotalGold', 'redTotalGold', 'blueWins', "Master Gold Graph")

    # Showing the Linear-Regression of total gold against wards, kills, healing, and object damage.

    columns = ['blueWardPlaced','blueKills','blueTotalHeal','blueObjectDamageDealt']

    # Challenger data
    results,x,y = get_linear_reg(chall_df_clean, columns, 'blueTotalGold')
    plot_linear_reg(results,x,y, title="Challenger Homoscedasticity", filename1="images/chall_homo_graph.png", filename2="images/chall_qq_plot.png")

    # # Grand Master data
    # results,x,y = get_linear_reg(gm_df_clean, columns, 'blueTotalGold')
    # plot_linear_reg(results,x,y, title="Grand Master Homoscedasticity", filename1="images/gm_homo_graph.png", filename2="images/gm_qq_plot.png")

    # # Master data
    # results,x,y = get_linear_reg(m_df_clean, columns, 'blueTotalGold')
    # plot_linear_reg(results,x,y, title="Master Homoscedasticity", filename1="images/m_homo_graph.png", filename2="images/m_qq_plot.png")

    # Getting sets of 25 graphs for each tier of play to create GIF's showing how over-investment in vision
    # can be detrimental to a team's win-rate.
    # columns = ['blueWins','blueWardPlaced', 'redWardPlaced']

    # make_gif_of_graphs(chall_df_clean, columns, 'blueWardPlaced', 'redWardPlaced', 2,filename='c_')
    # make_gif_of_graphs(gm_df_clean, columns, 'blueWardPlaced', 'redWardPlaced', 2, filename='gm_')
    # make_gif_of_graphs(m_df_clean, columns, 'blueWardPlaced', 'redWardPlaced', 2, filename='m_')

    # columns = ['blueWins','blueWardkills', 'redWardkills']

    # make_gif_of_graphs(chall_df_clean, columns, 'blueWardkills', 'redWardkills', 2, filename='c', scal=5)
    # make_gif_of_graphs(gm_df_clean, columns, 'blueWardkills', 'redWardkills', 2, filename='gm', scal=5)
    # make_gif_of_graphs(m_df_clean, columns, 'blueWardkills', 'redWardkills', 2, filename='m', scal=5)

    # # Hypothesis Testing Code - Challenger
    # hyp_test_df = chall_df_clean[(chall_df_clean['blueFirstDragon'] > 0) & (chall_df_clean['blueFirstTower'] == 0)]
    # hyp_test_df2 = chall_df_clean[(chall_df_clean['blueFirstTower'] > 0) & (chall_df_clean['blueFirstDragon'] == 0)]

    # drag_total = len(hyp_test_df)
    # tow_total = len(hyp_test_df2)
    # drag_sample_freq = np.sum(hyp_test_df['blueWins'])/drag_total
    # print(f"First dragon win frequency: {drag_sample_freq:2.2f}")
    # tow_sample_freq = np.sum(hyp_test_df2['blueWins'])/tow_total
    # print(f"First tower win frequency: {tow_sample_freq:2.2f}")
    # print(f"First Dragon win frequency: {drag_sample_freq}, First Tower win frequency: {tow_sample_freq}")
    # difference_in_sample_proportions = tow_sample_freq - drag_sample_freq
    # print("Difference in sample proportions: {:2.2f}".format(difference_in_sample_proportions))

    # # Hypothesis Testing Code - Grand Master
    # hyp_test_df = gm_df_clean[(gm_df_clean['blueFirstDragon'] > 0) & (gm_df_clean['blueFirstTower'] == 0)]
    # hyp_test_df2 = gm_df_clean[(gm_df_clean['blueFirstTower'] > 0) & (gm_df_clean['blueFirstDragon'] == 0)]

    # drag_total = len(hyp_test_df)
    # tow_total = len(hyp_test_df2)
    # drag_sample_freq = np.sum(hyp_test_df['blueWins'])/drag_total
    # print(f"First dragon win frequency: {drag_sample_freq:2.2f}")
    # tow_sample_freq = np.sum(hyp_test_df2['blueWins'])/tow_total
    # print(f"First tower win frequency: {tow_sample_freq:2.2f}")
    # print(f"First Dragon win frequency: {drag_sample_freq}, First Tower win frequency: {tow_sample_freq}")
    # difference_in_sample_proportions = tow_sample_freq - drag_sample_freq
    # print("Difference in sample proportions: {:2.2f}".format(difference_in_sample_proportions))

    # # Hypothesis Testing Code - Master
    # hyp_test_df = m_df_clean[(m_df_clean['blueFirstDragon'] > 0) & (m_df_clean['blueFirstTower'] == 0)]
    # hyp_test_df2 = m_df_clean[(m_df_clean['blueFirstTower'] > 0) & (m_df_clean['blueFirstDragon'] == 0)]

    # drag_total = len(hyp_test_df)
    # tow_total = len(hyp_test_df2)
    # drag_sample_freq = np.sum(hyp_test_df['blueWins'])/drag_total
    # print(f"First dragon win frequency: {drag_sample_freq:2.2f}")
    # tow_sample_freq = np.sum(hyp_test_df2['blueWins'])/tow_total
    # print(f"First tower win frequency: {tow_sample_freq:2.2f}")
    # print(f"First Dragon win frequency: {drag_sample_freq}, First Tower win frequency: {tow_sample_freq}")
    # difference_in_sample_proportions = tow_sample_freq - drag_sample_freq
    # print("Difference in sample proportions: {:2.2f}".format(difference_in_sample_proportions))