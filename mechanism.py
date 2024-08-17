import numpy as np
import torch
# device0='cuda:0'
# device1='cuda:1'
def agg_dp(x,ep,sen):
    delta = 1e-5
    sensitivity = sen
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / ep
    print(sigma)
    # 生成高斯噪声
    noise = torch.normal(mean=0, std=sigma, size=x.shape)
    noise=noise.to(x.device)
    # 添加噪声到张量
    noisy_tensor = x+ noise
    return noisy_tensor


import torch.distributions as dist

import torch
import torch.distributions as dist

def sample_from_fz(m, Sigma, epsilon, lambda_param):
    # Step 1: Sample N
    N = dist.MultivariateNormal(torch.zeros(m), torch.eye(m)).sample()

    # Step 2: Normalize X
    X = N / torch.norm(N, p=2)

    # Step 3: Sample Y from Gamma distribution
    gamma_dist = dist.Gamma(m, 1/epsilon)
    Y = gamma_dist.sample()

    # Step 4: Calculate Z
    I_m = torch.eye(m)
    Z = Y * torch.matmul((lambda_param * Sigma + (1 - lambda_param) * I_m) ** 0.5, X)

    return Z

def mahalanobis_mechanism(embeddings, Sigma, epsilon, lambda_param):
    perturbed_embeddings = []

    for emb in embeddings:
        # Step 3: Sample Z
        Z = sample_from_fz(emb.shape[0], Sigma, epsilon, lambda_param)
        
        # Step 4: Perturb the embedding
        perturbed_emb = emb + Z
        perturbed_embeddings.append(perturbed_emb)
    perturbed_embeddings=torch.tensor(perturbed_embeddings)
    return perturbed_embeddings

def mahalanobis_mechanism(emb, W, Sigma, epsilon, lambda_param):
    perturbed_s = []

    for i in emb.shape[0]:
        # Step 3: Sample Z
        Z = sample_from_fz(Sigma.shape[0], Sigma, epsilon, lambda_param)
        
        # Step 4: Perturb the embedding
        phi_i_hat = phi(w_i) + Z
        
        # Step 5: Replace w_i with closest embedding
        w_i_hat = min(W, key=lambda w: torch.norm(phi(w) - phi_i_hat, p=2))
        perturbed_s.append(w_i_hat)

    return perturbed_s