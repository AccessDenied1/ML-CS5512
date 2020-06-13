#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[141]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
X_1 = np.random.multivariate_normal(mean=[4, 0], cov=[[1, 0], [0, 1]], size=75)

X_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[2, 0], [0, 2]], size=250)

X_3 = np.random.multivariate_normal(mean=[1, 5], cov=[[1, 0], [0, 2]], size=20)

plt.scatter(X_1[:,0],X_1[:,1] , label = "X_1")
plt.scatter(X_2[:,0],X_2[:,1], label = "X_2")
plt.scatter(X_3[:,0],X_3[:,1], label = "X_3")
plt.legend(loc='lower right')
plt.show()
X =  np.concatenate((X_1,X_2,X_3))

# K-Means
print("K-Means")
sse = []
list_k = list(range(1, 10))
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
# Plot sse against k
print("Sum of squared distance = ",sse)
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters k')
plt.ylabel('Sum of squared distance');
plt.show()


n_iter = 9
fig, ax = plt.subplots(3, 3, figsize=(16, 16))
ax = np.ravel(ax)
centers = []
mi = 100000
for i in range(n_iter):
    # Run local implementation of kmeans
    km = KMeans(n_clusters=2,
                max_iter=100,
                random_state=np.random.randint(0, 1000))
    km.fit(X)
    centroids = km.cluster_centers_
    centers.append(centroids)
    ax[i].scatter(X[km.labels_ == 0, 0], X[km.labels_ == 0, 1],
                  c='green', label='cluster 1')
    ax[i].scatter(X[km.labels_ == 1, 0], X[km.labels_ == 1, 1],
                  c='blue', label='cluster 2')
    ax[i].scatter(centroids[:, 0], centroids[:, 1],
                  c='r', marker='*', s=300, label='centroid')
    ax[i].legend(loc='lower right')
    ax[i].set_aspect('equal')
    s = km.inertia_
    st = "sum of squared distance = "+str(s)
    ax[i].text(0,11.1,st, style='italic',)
    if(s < mi):
        fin_lab = km.labels_
        mi = s
        fin_cen = centroids
plt.tight_layout();
plt.show()
print("So the minimum squared distance is ",mi)
print("Hence the final clusters is ")
plt.scatter(X[fin_lab == 0, 0], X[fin_lab == 0, 1],
                  c='green', label='cluster 1')
plt.scatter(X[fin_lab == 1, 0], X[fin_lab == 1, 1],
                  c='blue', label='cluster 2')
plt.scatter(fin_cen[:, 0], fin_cen[:, 1],
                  c='r', marker='*', s=300, label='centroid')
plt.legend(loc='lower right')
plt.show()
print("Centre =",fin_cen)


#GMM
print("GMM")
##GMM

from scipy.stats import multivariate_normal
def plot_contours(data, means, covs, title):
    from matplotlib.pyplot import figure
    plt.figure(figsize=(7,5))
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors = col[i])
        #plt.xlim(0,7)
        #plt.ylim(-3,4)
    plt.title(title)
    plt.tight_layout()
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


sse = []
list_k = list(range(1, 10))
for k in list_k:
    gmm = GMM(n_components=k)
    gmm.fit(X)
    sse.append(-1*gmm.score(X))
# Plot sse against k
#print("Sum of squared distance = ",sse)
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of component k')
plt.ylabel('-1*Log likelihood of X');
plt.show()


n_components = np.arange(1,10)
clfs = [GMM(n_components = n, max_iter = 1000).fit(X) for n in n_components]
bics = [clf.bic(X) for clf in clfs]
aics = [clf.aic(X) for clf in clfs]

plt.plot(n_components, bics, label = 'BIC')
plt.plot(n_components, aics, label = 'AIC')
plt.xlabel('n_components')
plt.legend()
plt.show()

gmm = GMM(n_components = 3)
gmm.fit(X)
print("Mean = " , gmm.means_)
plot_contours(X, gmm.means_, gmm.covariances_, 'Final Clusters')
plt.show()
plot_gmm(gmm,X)
plt.show()



#Image compression

# Read the image
img = imread('iitpkd.jpg')
img_size = img.shape

# Reshape it to be 2-dimension
X = img.reshape(img_size[0] * img_size[1], img_size[2])

# Run the Kmeans algorithm
km = KMeans(n_clusters=30)
km.fit(X)

# Use the centroids to compress the image (Using cluster center as the pixel values)
X_compressed = km.cluster_centers_[km.labels_]
X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)

# Reshape X_recovered to have the same dimension as the original image
X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])

# Plot the original and the compressed image next to each other
fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(X_compressed)
ax[1].set_title('Compressed Image with 30 colors')
for ax in fig.axes:
    ax.axis('off')
plt.tight_layout();
plt.show()
img = Image.fromarray(X_compressed)
img.save('compressed.jpg')


# In[ ]:




