import numpy as np
from sklearn.linear_model import LinearRegression

n = 120
beta = np.array([0.5, 2.5, -1.44])
p = np.shape(beta)[0]

mu = 0
sigma = 0.098765
seed = 4321
rng = np.random.default_rng(seed)
epsilon = rng.normal(mu, sigma, n)

x1 = np.ones((n,1))
x2 = rng.random((n,1))
x3 = np.repeat([1,2,3,4,5], n/5)[:,np.newaxis]
x = np.concatenate((x1,x2,x3), axis=1)
y = x @ beta + epsilon

reg = LinearRegression().fit(x, y)
true_beta_hat = np.append(reg.intercept_, reg.coef_[1:p])

def compute_gamma_val(regressor, regressand):
    return (regressor.T @ regressand) / (regressor.T @ regressor)

def compute_gamma_and_residuals(x):
    n,p = np.shape(x)
    gamma = np.diag(np.ones(p))
    Z = np.empty((n,p))
    Z[:,0] = np.ones(n)
    for j_idx in range(1,p):
        for k_idx in range(p):
            if k_idx < j_idx:
                gamma[k_idx, j_idx] = compute_gamma_val(Z[:, k_idx], x[:, j_idx])
            else:
                pass
    Z[:,j_idx] = x[:,j_idx] - sum([gamma[i_idx,j_idx] * Z[:,i_idx] for i_idx in range(j_idx)])
    return (gamma, Z)

def compute_D(residuals):
    _,p = np.shape(residuals)
    norms = np.array([np.linalg.norm(residuals[:,col_idx]) for col_idx in range(p)])
    D = np.diag(norms)
    return D

def gram_schmidt(x,y):
    gamma, Z = compute_gamma_and_residuals(x)
    D = compute_D(Z)
    Q = Z @ np.linalg.inv(D)
    R = D @ gamma
    beta_hat = np.linalg.solve(R, Q.T @ y)
    return beta_hat

print("Beta hat from sklearn: ", true_beta_hat)
print("Beta hat from our G-S procedure:", gram_schmidt(x,y))
