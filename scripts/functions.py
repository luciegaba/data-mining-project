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

def missing_rate_report(X_train):
    missing_rate=pd.DataFrame({'count': X_train.isna().sum(),
                             'rate': (X_train.isna().sum()*100/X_train.shape[0])}).sort_values(by = 'rate', ascending = False)
    return missing_rate



##### PCA & CLUSTERING #####

 
def pca_graph(X_normalise: np.array,geo_variables_for_label:pd.Series):
    acp = PCA(n_components=2)
    acp.fit(X_normalise) 
    df_graph=pd.concat([pd.DataFrame(acp.transform(X_normalise)),geo_variables_for_label],axis=1)
    fig = px.scatter(df_graph, x=0, y=1,text=geo_variables_for_label.name,color_discrete_sequence=['rgb(94, 201, 98,0.5)'])
    fig.update_traces(textposition="bottom center")
    fig.update_layout(showlegend=False,autosize = True,hovermode='closest', width=2000,
    height=2000,)
    fig.show()



 
def pca_graph_clusters(X_normalise: np.array,geo_variables_for_label:pd.Series,preds:np.array):
    acp = PCA(n_components=2)
    acp.fit(X_normalise) 
    df_graph=pd.concat([pd.DataFrame(acp.transform(X_normalise)),geo_variables_for_label,pd.Series(preds,name='preds')],axis=1)
    fig = px.scatter(df_graph, x=0, y=1,text=geo_variables_for_label.name,color_discrete_sequence=['rgb(94, 201, 98,0.5)'],color="preds")
    fig.update_traces(textposition="bottom center")
    fig.update_layout(showlegend=False,autosize = True,hovermode='closest', width=1000,height=1000,)
    fig.show()



def get_correlation_circle_2_components(X_normalise:pd.DataFrame,X_raw:pd.DataFrame):
    n,p = X_normalise.shape # nb individus  # nb variables
    acp=PCA()
    acp.fit(X_normalise)
    eigval = ((n-1) / n) * acp.explained_variance_ # valeurs propres
    sqrt_eigval = np.sqrt(eigval) # racine carrée des valeurs propres
    fig, ax = plt.subplots(figsize=(30, 30))
    texts = []
    for i in range(0, acp.components_.shape[1]):
        ax.arrow(0,
                0,  # Start the arrow at the origin
                acp.components_[0, i]*sqrt_eigval[0],  #0 for PC1
                acp.components_[1, i]*sqrt_eigval[1],  #1 for PC2
                head_width=0.01,
                head_length=0.01, 
                width=0.001,              )
        
        random_position_x = random.choice(np.arange(0,0.1,0.02))
        random_position_y = random.choice(np.arange(0,0.1,0.02))
        plt.text(acp.components_[0, i]*sqrt_eigval[0]+random_position_x,acp.components_[1, i]*sqrt_eigval[1]+random_position_y,X_raw.iloc[:,i].name)


    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')


    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(1, round(100*acp.explained_variance_ratio_[0],1)))
    plt.ylabel('F{} ({}%)'.format(2, round(100*acp.explained_variance_ratio_[1],1)))

    plt.title("Correlations circle (F{} et F{})".format(1, 2))


    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    plt.show(block=False)

def features_corr_for_2_components(X_normalise,components_df,X_raw):
    correlations_axe1 = []
    correlations_axe2 = []

    for i,feat in enumerate(X_raw.columns):
        corr_axe1 = np.corrcoef(X_normalise[:,i],components_df['pca_1'].values)[0][1]
        corr_axe2 = np.corrcoef(X_normalise[:,i],components_df['pca_2'].values)[0][1]
        correlations_axe1.append(corr_axe1)
        correlations_axe2.append(corr_axe2)
        
    df_coord = pd.DataFrame({'id': X_raw.columns, 'corr_axe1': correlations_axe1, 'corr_axe2': correlations_axe2})
    return df_coord



def kmeans_graph(X_normalise:pd.DataFrame):
    score = []

    for i in range(2,15+1):
        km = KMeans(n_clusters=i,random_state=0).fit(X_normalise)
        preds = km.predict(X_normalise)
        score.append(-km.score(X_normalise))
    plt.figure(figsize=(12,5))
    plt.title("Elbow method",fontsize=16)
    plt.plot(range(2,16),score,marker='o')
    plt.grid(True)
    plt.xlabel('Number of clusters',fontsize=14)
    plt.ylabel('K-means score',fontsize=14)
    plt.xticks(range(2,16),fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()


##### MODELING #####




def create_mod_for_secretisees(maille):
    if maille ==0:
        maille="grp_1"
    elif maille ==1:
        maille="grp_2"
    else:
        maille="grp_3"
    return maille




def erase_correlated_var(X):
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    print("High correlated variables:",to_drop)
    return to_drop


def processing_ml(base_electricite,target,scaler=False):
    X=base_electricite.drop(columns=[target,"code_commune"])
    print(X.columns)
    y=base_electricite[target]
    if scaler == True:
        quant_features=[var for var in X.columns if (not var.startswith("nombre")) or (not var.startswith("cluster"))]
        scaler=MinMaxScaler()
        X[quant_features]=scaler.fit_transform(X[quant_features])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
    return X_train,X_test,y_train,y_test
    


def grid_search_ridge(X, y):
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    ridge = Ridge()
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)
    print("Best alpha: {}".format(grid_search.best_params_))
    return grid_search.best_params_




def OLS_reports(X_train,X_test,y_train,y_test,target,ridge=True):
    mod = sm.OLS(y_train.values,X_train)
    ols = mod.fit()
    print(f"OLS results for {target}")
    print(ols.summary())

    if ridge==True:
        alpha=grid_search_ridge(X_train, y_train)
        linear_reg=Ridge(**alpha)
    else:   
        linear_reg=LinearRegression()
        
    linear_reg.fit(X_train,y_train)
    y_train_pred=linear_reg.predict(X_train)
    y_test_pred=linear_reg.predict(X_test)
    print("MAE on train: ",metrics.mean_absolute_error(y_train,y_train_pred))
    print("MAE on test:",metrics.mean_absolute_error(y_test,y_test_pred))
    print()
    print("MedAE on train: ",metrics.median_absolute_error(y_train,y_train_pred))
    print("MedAE on test: ",metrics.median_absolute_error(y_train,y_train_pred))

    return linear_reg