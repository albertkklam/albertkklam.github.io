import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

n = 200
beta = np.array([1.5, 2.5])
p = np.shape(beta)[0]

mu = 0
sigma_1 = 0.123
seed = 456
rng = np.random.default_rng(seed)
epsilon = rng.normal(mu, sigma_1, n)

x0 = np.ones((n,1))
x1 = rng.random((n,1))
x = np.concatenate((x0,x1), axis=1)
y = x @ beta + epsilon

reg = LinearRegression().fit(x,y)
y_pred = reg.predict(x)
beta_hat = np.append(reg.intercept_,reg.coef_[1:p])

data = np.concatenate((x1,y[...,np.newaxis]), axis=1)
pca = PCA().fit(data)
pca_data = pca.transform(data)
pca_components = pca.components_
pca_mean = pca.mean_
pc1_data = np.dot(pca_data[:,0].reshape(-1,1), pca_components[:,0].reshape(1,-1)) + pca_mean
pc1_x1, pc1_y = pc1_data[:,0], pc1_data[:,1]

plt.figure()
plt.scatter(x1, y, s=5, color="black")
plt.plot(x1, y_pred, color="red", linewidth=1)
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, 5.5, step=0.5))
plt.xlabel(r"Data ($X_1$)")
plt.ylabel(r"Response (Y)")
plt.title(r"Linear Regression of $Y$ on $X$")
os.chdir("/Users/Albert/Personal/Projects/blog/content/posts/images/")
plt.savefig("reg.png")

plt.figure()
plt.scatter(x1, y, s=5, color="black")
plt.plot(x1, y_pred, color="red", linewidth=1)
plt.plot(pc1_x1, pc1_y, color="blue", linewidth=1)
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, 5.5, step=0.5))
plt.xlabel(r"Data ($X_1$)")
plt.ylabel(r"Response (Y)")
plt.title(r"Linear Regression on $X$ and the First PC of $(X_1, Y)$")
plt.savefig("reg_and_pca.png")

abs_diff = np.abs(y_pred - pc1_y)

plt.figure()
plt.scatter(x1, abs_diff, s=5, color="black")
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.xlabel(r"Data ($X_1$)")
plt.ylabel(r"Absolute Difference")
plt.title(r"Absolute Difference between Regression and PCA")
plt.savefig("abs_diff.png")
