import numpy as np



class Multiple_Dimensional_Scaling:
    def euclidean_dist(self,x1,x2):
        """
        Parameters
        ----------
        x1 : 1-D array, feature vector of instance.
        x2 : 1-D array, feature vector of instance.
        Returns
        -------
        dist: float,distance of x1 and x2.
        """
        dist = np.linalg.norm(x1-x2)
        return dist
    def get_dist_matrix(self,data):
        """
        Parameters
        ----------
        data : 2-D array, feature space
        Returns
        -------
        dist_matrix: 2-D array, distances between instances.
        """
        num = data.shape[0]
        dist_matrix = np.zeros((num,num),dtype=float)
        for i in range(num):
            for j in range(num):
                dist_matrix[i,j]=self.euclidean_dist(data[i], data[j])
        return dist_matrix
    def get_inner_prod_matrix(self,dist_matrix):
        """
        Parameters
        ----------
        dist_matrix : 2-D array, dist matrix from get_dist_matrix function.
        Returns
        -------
        inner_prod_matrix: 2-D array,inner product of low space.
        """
        D_square = np.square(dist_matrix)
        D_sum = np.sum(D_square,axis=1)/D_square.shape[0]
        D_i = np.repeat(D_sum[:,np.newaxis], dist_matrix.shape[0],axis=1)
        D_j = np.repeat(D_sum[np.newaxis,:], dist_matrix.shape[0],axis=0)
        D_ij = np.sum(D_square)/((dist_matrix.shape[0])**2)*np.ones([dist_matrix.shape[0],dist_matrix.shape[0]])
        inner_prod_matrix = (D_i + D_j - D_square - D_ij)/2
        return inner_prod_matrix
    def fit(self,data, low_dim =2):
        D = self.get_dist_matrix(data)
        B = self.get_inner_prod_matrix(D)
        eig_values, eig_vects = np.linalg.eigh(B)
        eig_values_sort = np.argsort(-eig_values)
        values_sort = eig_values[eig_values_sort]
        vects_sort = eig_vects[:,eig_values_sort]
        top_value_diag = np.diag(values_sort[0:low_dim])
        top_vects = vects_sort[:,0:low_dim]
        Z = np.dot(np.sqrt(top_value_diag),top_vects.T).T
        return Z


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from sklearn.manifold import MDS
from sklearn.datasets import make_swiss_roll
X, y = make_swiss_roll(n_samples=5000,noise=0.2,random_state=42)

mds = Multiple_Dimensional_Scaling()
Z = mds.fit(X,low_dim=2)

mds_sklearn = MDS(n_components=2,metric=True)
X_mds = mds_sklearn.fit_transform(X)

axes =[-11.5,14,-2,23,-12,15]
fig = plt.figure()
ax1 = fig.add_subplot(131,projection="3d")
ax1.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.hot)
ax1.view_init(10,60)
ax1.set_xlabel("$x$",fontsize=18)
ax1.set_ylabel("$y$",fontsize=18)
ax1.set_zlabel("$z$",fontsize=18)
ax1.set_xlim(axes[0:2])
ax1.set_ylim(axes[2:4])
ax1.set_zlim(axes[4:6])
plt.title("3-D Swiss Roll")
ax2 = fig.add_subplot(132)
ax2.scatter(Z[:,0],Z[:,1],c=y,cmap=plt.cm.hot)
ax2.set_xlabel("$x$",fontsize=18)
ax2.set_ylabel("$y$",fontsize=18)
plt.title(" after MDS")
ax3 = fig.add_subplot(133)
ax3.scatter(X_mds[:,0],X_mds[:,1],c=y,cmap=plt.cm.hot)
ax3.set_xlabel("$x$",fontsize=18)
ax3.set_ylabel("$y$",fontsize=18)
plt.title(" after MDS in sklearn")
plt.show()