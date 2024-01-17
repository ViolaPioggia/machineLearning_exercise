import numpy as np
from matplotlib import pyplot as plt


class Priceipal_Component_Analysis:
    def fit(self, data, dim):
        """
        Parameters
        ----------
        data : 2-D numpy array, original feature space
        dim : int, dimension of low feature space
        """
        # data normalization
        mean_values = np.mean(data,axis=0)
        data_centered = data - mean_values
        # calculate cov
        data_cov = np.cov(data_centered,rowvar=0)
        # eigen-decomposition
        eig_values, eig_vects = np.linalg.eig(data_cov)
        # sort eig_values
        eig_values_sorted = np.argsort(-eig_values)
        #select biggest dim eigvalues
        top_dim_index = eig_values_sorted[:dim]
        # select top dim eig vectors
        top_dim_vects = eig_vects[:,top_dim_index]
        self.w = top_dim_vects
    def transform(self,data):
        """
        Parameters
        ----------
        data : 2-D numpy array, original feature space
        Returns
        -------
        data_low: 2-D numpy array, PCA of data
        """
        # data normalization
        mean_values = np.mean(data,axis=0)
        data_centered = data - mean_values
        data_low = np.dot(data_centered,self.w)
        return data_low
    def fit_transform(self,data,dim):
        """
        Parameters
        ----------
        data : 2-D numpy array, original feature space
        dim : int, dimension of low feature space
        Returns
        ----------
        data_low, 2-D numpy array, PCA of data
        """
        # data normalization
        mean_values = np.mean(data,axis=0)
        data_centered = data - mean_values
        # calculate cov
        data_cov = np.cov(data_centered,rowvar=0)
        # eigen-decomposition
        eig_values, eig_vects = np.linalg.eig(data_cov)
        # sort eig_values
        eig_values_sorted = np.argsort(-eig_values)
        #select biggest dim eigvalues
        top_dim_index = eig_values_sorted[:dim]
        # select top dim eig vectors
        top_dim_vects = eig_vects[:,top_dim_index]
        self.w = top_dim_vects
        data_low = np.dot(data_centered,self.w)
        return data_low

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
data = cancer_data['data']#feature space
y = cancer_data['target']#label space

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data) # data normalization

# create figures
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
#show original data
x1 = np.argwhere(cancer_data['feature_names']=='mean symmetry')[0][0]
x2 = np.argwhere(cancer_data['feature_names']=='worst smoothness')[0][0]
x = data[:,[x1,x2]]
ax1.scatter(x[:,0],x[:,1],c=y,s=40,cmap=plt.cm.Spectral)

# do PCA tranformation
pca = Priceipal_Component_Analysis()
data_low = pca.fit_transform(data,dim=2)
# show low dimension PCA data
ax2.scatter(data_low[:,0],data_low[:,1],c=y,s=40,cmap=plt.cm.Spectral)

# do PCA using sklearn
from sklearn.decomposition import PCA
pca_sklearn = PCA(n_components=2)
reduced_data = pca_sklearn.fit_transform(data)
ax3.scatter(reduced_data[:,0],reduced_data[:,1],c=y,s=40,cmap=plt.cm.Spectral)

plt.show()