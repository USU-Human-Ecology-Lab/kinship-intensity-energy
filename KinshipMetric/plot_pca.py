import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white')
import pandas as pd

sns.set_style({'font.family' : 'serif'})


data_0_loadings = [0.22882388 ,0.11129247 ,0.01740337]
data_0_explaind = [0.6400315  ,0.31129043 ,0.04867806]

data_1_loadings = [0.18802306 ,0.10338041 ,0.02063583]
data_1_explaind = [0.60256211 ,0.33130573 ,0.06613215]

data_2_loadings = [0.17843857 ,0.09616163 ,0.02065378]
data_2_explaind = [0.60435619 ,0.32569122 ,0.06995259]

data_3_loadings = [0.11036392 ,0.04173463 ,0.02596319]
data_3_explaind = [0.61980703 ,0.23438291 ,0.14581006]

loadings = pd.DataFrame(
        [data_0_loadings,data_1_loadings,data_2_loadings,data_3_loadings],
        columns=['PC1','PC2','PC3']
        )

explaind = pd.DataFrame(
        [data_0_explaind, data_1_explaind, data_2_explaind, data_3_explaind],
        columns=[1,2,3]
        )

sns.set_palette('colorblind')
ax = sns.lineplot(data=explaind.transpose(),
        lw=2.5,
        ms=14,
        markers=['^','o','*','s']
        )
ax.legend(labels=['0 allowed','1 allowed','2 allowed', '3 allowed'])
ax.set(xticks=[1,2,3])
ax.yaxis.grid(True)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.show()


#loadings.index = ['0 allowed','1 allowed','2 allowed','3 allowed']
#
#data = []
#for i in loadings.index:
#    for pc in loadings.columns:
#        data.append({ 
#            'x' : pc,
#            'y' : loadings.at[i, pc],
#            'Imputation treatment' : i
#        })
#        print(loadings.at[i, pc])
#
#data = pd.DataFrame(data)
#ax = sns.catplot(data=data, x='x', y='y', hue='Imputation treatment', kind='bar')
#plt.xlabel('Principal component')
#plt.ylabel('Loading')
#plt.show()
