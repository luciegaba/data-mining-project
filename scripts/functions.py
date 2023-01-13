import pandas as pd
import numpy as np
import random
import statsmodels.api as sm


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics

import matplotlib.pyplot as plt
import plotly.express as px








##### EXPLORATORY ANALYSIS #####

def missing_rate_report(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of training data, returns a DataFrame containing the count and rate of missing values for each column.
    Columns are sorted in descending order of missing rate.
    """
    missing_rate = pd.DataFrame({'count': X_train.isna().sum(),
                             'rate': (X_train.isna().sum()*100/X_train.shape[0])}).sort_values(by = 'rate', ascending = False)
    return missing_rate

##### PCA & CLUSTERING #####

 
def pca_graph(X_normalise: np.ndarray, geo_variables_for_label: pd.Series) -> None:
    """
    Given a 2D numpy array of normalized data and a pandas Series of labels,
    performs PCA on the normalized data and plots a scatter plot with the resulting principal components,
    using the provided labels as text on the plot.
    """
    acp = PCA(n_components=2)
    acp.fit(X_normalise) 
    df_graph=pd.concat([pd.DataFrame(acp.transform(X_normalise)),geo_variables_for_label],axis=1)
    fig = px.scatter(df_graph, x=0, y=1,text=geo_variables_for_label.name,color_discrete_sequence=['rgb(94, 201, 98,0.5)'])
    fig.update_traces(textposition="bottom center")
    fig.update_layout(showlegend=False,autosize = True,hovermode='closest', width=2000,
    height=2000,)
    fig.show()



 
def pca_graph_clusters(X_normalise: np.ndarray, geo_variables_for_label: pd.Series, preds: np.ndarray) -> None:
    """
    Given a 2D numpy array of normalized data, a pandas Series of labels, and a 1D numpy array of cluster predictions,
    performs PCA on the normalized data and plots a scatter plot with the resulting principal components,
    using the provided labels as text on the plot and color-coding the points by their cluster predictions.
    """
    acp = PCA(n_components=2)
    acp.fit(X_normalise) 
    df_graph=pd.concat([pd.DataFrame(acp.transform(X_normalise)),geo_variables_for_label,pd.Series(preds,name='preds')],axis=1)
    fig = px.scatter(df_graph, x=0, y=1,text=geo_variables_for_label.name,color_discrete_sequence=['rgb(94, 201, 98,0.5)'],color="preds")
    fig.update_traces(textposition="bottom center")
    fig.update_layout(showlegend=False,autosize = True,hovermode='closest', width=1000,height=1000,)
    fig.show()

    
def get_correlation_circle_2_components(X_normalise: pd.DataFrame, X_raw: pd.DataFrame):
    """
    This function creates a correlation circle plot of the first two principal components of the data in X_normalise.
    The function also adds the names of the variables from X_raw to the plot.
    Inputs:
    - X_normalise: a dataframe of normalized data
    - X_raw: a dataframe of raw data, with variable names to be added to the plot
    """
    n, p = X_normalise.shape  # nb individus  # nb variables
    acp = PCA()
    acp.fit(X_normalise)
    eigval = ((n - 1) / n) * acp.explained_variance_  # valeurs propres
    sqrt_eigval = np.sqrt(eigval)  # racine carrée des valeurs propres
    fig, ax = plt.subplots(figsize=(30, 30))
    texts = []
    for i in range(0, acp.components_.shape[1]):
        ax.arrow(
            0,
            0,  # Start the arrow at the origin
            acp.components_[0, i] * sqrt_eigval[0],  # 0 for PC1
            acp.components_[1, i] * sqrt_eigval[1],  # 1 for PC2
            head_width=0.01,
            head_length=0.01,
            width=0.001,
        )

        random_position_x = random.choice(np.arange(0, 0.1, 0.02))
        random_position_y = random.choice(np.arange(0, 0.1, 0.02))
        plt.text(
            acp.components_[0, i] * sqrt_eigval[0] + random_position_x,
            acp.components_[1, i] * sqrt_eigval[1] + random_position_y,
            X_raw.iloc[:, i].name,
        )

    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color="grey", ls="--")
    plt.plot([0, 0], [-1, 1], color="grey", ls="--")

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel(
        "F{} ({}%)".format(1, round(100 * acp.explained_variance_ratio_[0], 1))
    )
    plt.ylabel(
        "F{} ({}%)".format(2, round(100 * acp.explained_variance_ratio_[1], 1))
    )

    plt.title("Correlations circle (F{} et F{})".format(1, 2))
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    plt.show(block=False)
                                                        
                                                        


def features_corr_for_2_components(X_normalise, components_df, X_raw):
    """
    This function calculates the correlation between each feature in X_normalise and the first two principal components in components_df.
    The function returns a dataframe with the feature names, their correlation with the first component, and their correlation with the second component.
    Inputs:
    - X_normalise: a dataframe of normalized data
    - components_df: a dataframe of principal component scores, with columns 'pca_1' and 'pca_2'
    - X_raw: a dataframe of raw data, with feature names to be added to the output dataframe
    """
    correlations_axe1 = []
    correlations_axe2 = []

    for i, feat in enumerate(X_raw.columns):
        corr_axe1 = np.corrcoef(X_normalise[:, i], components_df["pca_1"].values)[0][1]
        corr_axe2 = np.corrcoef(X_normalise[:, i], components_df["pca_2"].values)[0][1]
        correlations_axe1.append(corr_axe1)
        correlations_axe2.append(corr_axe2)

    df_coord = pd.DataFrame(
        {
            "id": X_raw.columns,
            "corr_axe1": correlations_axe1,
            "corr_axe2": correlations_axe2,
        }
    )
    return df_coord



def kmeans_graph(X_normalise: pd.DataFrame):
    """
    This function plots the k-means score against the number of clusters, using the Elbow Method, to help select the optimal number of clusters for k-means clustering on the data in X_normalise.
    Inputs:
    - X_normalise: a dataframe of normalized data
    """
    score = []

    for i in range(2, 15 + 1):
        km = KMeans(n_clusters=i, random_state=0).fit(X_normalise)
        preds = km.predict(X_normalise)
        score.append(-km.score(X_normalise))
    plt.figure(figsize=(12, 5))
    plt.title("Elbow method", fontsize=16)
    plt.plot(range(2, 16), score, marker="o")
    plt.grid(True)
    plt.xlabel("Number of clusters", fontsize=14)
    plt.ylabel("K-means score", fontsize=14)
    plt.xticks(range(2, 16), fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()



##### MODELING #####



def create_mod_for_secretisees(maille: int):
    """
    This function takes an integer input and maps it to one of three string values based on its value, using if-else statements.
    The function is used to create modalities for secretisees.
    Input:
    - maille: an integer, can be 0, 1, or any other integer
    Output:
    - maille: a string, can be "grp_1", "grp_2", or "grp_3"
    """
    if maille == 0:
        maille = "grp_1"
    elif maille == 1:
        maille = "grp_2"
    else:
        maille = "grp_3"
    return maille

def erase_correlated_var(X:pd.DataFrame):
    """
    This function takes a pandas dataframe as an input, and returns a list of highly correlated variables based on a threshold of 0.9.
    Input:
    - X: a pandas dataframe of features
    Output:
    - to_drop: a list of the highly correlated variables
    """
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    print("High correlated variables:", to_drop)
    return to_drop


def processing_ml(base_electricite: pd.DataFrame, target: str, scaler: bool = False):
    """
    This function takes a dataframe of electric consumption data, a target variable, and an optional scaler flag. 
    It returns train and test splits of the dataframe and target variable, after dropping certain columns and applying MinMaxScaler to the quantitive features if scaler is set to True.
    Inputs:
    - base_electricite: a dataframe of electric consumption data
    - target: a string of the target variable column name
    - scaler: a boolean flag for applying MinMaxScaler, default is False
    Outputs:
    - X_train, X_test, y_train, y_test: train and test splits of the input dataframe and target variable
    """
    X = base_electricite.drop(columns=[target, "code_commune"])
    print(X.columns)
    y = base_electricite[target]
    if scaler == True:
        quant_features = [
            var for var in X.columns if (not var.startswith("nombre")) or (not var.startswith("cluster"))
        ]
        scaler = MinMaxScaler()
        X[quant_features] = scaler.fit_transform(X[quant_features])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def grid_search_ridge(X: pd.DataFrame, y: pd.DataFrame):
    """
    This function performs a grid search on a Ridge model to find the best regularization parameter, and prints and returns the best parameter.
    Inputs:
    - X: a dataframe of features
    - y: a dataframe of target variable
    Outputs:
    - best_params: a dictionary containing the best regularization parameter as the value of key 'alpha'
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
    ridge = Ridge()
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print("Best alpha: {}".format(best_params))
    return best_params


def OLS_reports(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, target: str, ridge: bool = True):
    """
    This function performs OLS regression on the input data and prints summary statistics of the model fit, as well as Mean Absolute Error and Median Absolute Error on the train and test sets.
    Inputs:
    - X_train: a dataframe of training features
    - X_test: a dataframe of test features
    - y_train: a dataframe of training target variable
    - y_test: a dataframe of test target variable
    - target: a string of the target variable column name
    - ridge: a boolean flag for whether to use Ridge regression instead of Linear Regression, default is True
    Outputs:
    - linear_reg: a fitted Linear Regression or Ridge Regression model
    """
    import statsmodels.api as sm
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn import metrics
    mod = sm.OLS(y_train.values, X_train)
    ols = mod.fit()
    print(f"OLS results for {target}")
    print(ols.summary())

    if ridge == True:
        alpha = grid_search_ridge(X_train, y_train)
        linear_reg = Ridge(**alpha)
    else:
        linear_reg = LinearRegression()

    linear_reg.fit(X_train, y_train)
    y_train_pred = linear_reg.predict(X_train)
    y_test_pred = linear_reg.predict(X_test)
    print("MAE on train: ", metrics.mean_absolute_error(y_train, y_train_pred))
    print("MAE on test:", metrics.mean_absolute_error(y_test, y_test_pred))
    print()
    print("MedAE on train: ", metrics.median_absolute_error(y_train, y_train_pred))
    print("MedAE on test: ", metrics.median_absolute_error(y_test, y_test_pred))

    return linear_reg
