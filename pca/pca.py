import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white')
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from mpl_toolkits.mplot3d import Axes3D

# Use supernatural enforcement variables as a proxy for kinship intensity
COLS = [
        'Supernatural_enforcement_of_fairness',
        'Supernatural_enforcement_of_human_reciprocity',
        'Supernatural_enforcement_of_ingroup_loyalty',
        ]


def truncate(x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x

def compute_scores(X, n_components):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))
    return pca_scores, fa_scores

def freqToScore(freq):
    return {
        # Frequencies assigned scores according to yearly occurences
        'daily'                 : 365,
        'weekly'                : 52,
        'monthly'               : 12,
        'monthly/seasonal'      : 8,
        'seasonal'              : 4,
        'seasonally'            : 4,
        'yearly'                : 1,
        'once yearly'           : 1,
        'once in a generation'  : 1/25,
        'once per generation'   : 1/25,
        'once in a lifetime'    : 1/60
    }[freq]/365

# Check to see if a series meets the specified imputation threshold
def imputationThreshold(series, maxImputed):
    features_imputed = eval(series['Features_imputed'])
    num_imputed = np.sum([features_imputed[c] for c in COLS])
    return num_imputed <= maxImputed

# Restrict data to the specified imputation threshold
def restrict(df, maxImputed):
    threshold = lambda series : imputationThreshold(series, maxImputed)
    return df[COLS][df.apply(threshold, axis=1)].copy()

# Perform three tests to check the number of significant PCs
def testPCsignificance(X, n):
    n_components = [n for n in np.arange(0, len(COLS)+1)]
    pca_scores, fa_scores = compute_scores(X, n_components)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    X = X.copy()
    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("Best n_components by PCA CV = %d" % n_components_pca)
    print("Best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("Best n_components by PCA MLE = %d" % n_components_pca_mle)

def runPCA(X, n):
    X = X.copy()
    newData = X.copy()
    pca = PCA(svd_solver='full')
    pca.fit(X)
    components = pca.transform(X)

    print('Loadings: {}'.format(pca.explained_variance_))
    print('Explained variance: {}'.format(pca.explained_variance_ratio_))

    pcs = np.transpose(components)

    newData['PC1'] = pcs[0]
    newData['PC2'] = pcs[1]
    newData['PC3'] = pcs[2]
    newData['Kinship_index'] = newData['PC1'] + newData['PC2'] + newData['PC3']

    return newData

def mergeNewMetrics(baseData, newData):
    for metric in ['PC1','PC2','PC3','Kinship_index']:
        baseData[metric] = newData[metric]
    return baseData

def main():
    # Read in data
    data        = pd.read_csv('ShinySeshatLatLong.csv',index_col=0)
    dataVanilla = pd.read_csv('shiny-seshat.csv',index_col=0)
    data['Features_imputed'] = dataVanilla['Features_imputed']

    # Truncate values to the interval [0,1]
    for col in COLS:
        data[col] = data[col].apply(truncate)

    # Make four datasets for four imputation treatments
    for i in range(4):
        print('######################')
        print('### data_{}_imputed ###'.format(i))
        print('######################')

        # Restrict the dataset to only allow i imputed supernatural columns
        data_i_imputed = restrict(data, i)
        # Tell us the size
        print('Size : {} polity-centuries'.format(len(data_i_imputed)))
        # Print out the results of several PC significance tests
        testPCsignificance(data_i_imputed, i)
        # Run the actual PCA
        data_i_imputed = runPCA(data_i_imputed, i)
        # Merge the PC data to the base dataset
        full_data_i = mergeNewMetrics(data, data_i_imputed)
        # Export to CSV
        full_data_i.to_csv('seshat-kinship_allow-{}.csv'.format(i))
        print('')

#    pcs = pd.DataFrame(components, columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])

#    pcs['Aggregate_metric'] = pcs['PC1'] + pcs['PC2'] + pcs['PC3'] + pcs['PC4']
#    print(pcs['Aggregate_metric'])
#    sns.histplot(pcs['Aggregate_metric'],bins=20)
#    plt.show()

#    pcs.index = data.index
#    pcs.to_csv('PCA_kinship_metric.csv')

#    fig = plt.figure()
#    ax = Axes3D(fig)
#    pcs = pd.DataFrame(components)
#    print(pcs)
#    ax.scatter(pcs[0],pcs[1],pcs[2])
#    ax.set_xlabel('PC1')
#    ax.set_ylabel('PC3')
#    ax.set_zlabel('PC3')
#    plt.show()


if __name__ == '__main__':
    main()
#pca = PCA()
#pca.fit(data)
#components = pca.transform(data)
#
#print('Loadings [PC1 PC2 PC3]')
#print(pca.explained_variance_)
#print('Explained variance [PC1 PC2 PC3]')
#print(pca.explained_variance_ratio_)
#
#loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
#loading_matrix = pd.DataFrame(loadings, columns=['PC1','PC2','PC3'], index=COLS) #print(loading_matrix)
#
#pcs = (pd.DataFrame(components,columns=['PC1','PC2','PC3']))
#
#
#x = pcs['PC1']
#y = pcs['PC2']
#sns.kdeplot(x,y,cmap='plasma_r',shade=True,shade_lowest=False,bw=0.15, alpha = 1)
#plt.show()
#
