import torch
import torch.nn as nn

def gmm_score(x, means, covs, weights):
    """
    Compute the score function (gradient of log probability) for a Gaussian Mixture Model.
    
    ∇ log p(x) = ∑_k π_k N(x|μ_k,Σ_k) ∇ log N(x|μ_k,Σ_k) / p(x)
               = ∑_k w_k(x) Σ_k^{-1} (μ_k - x)
    
    where w_k(x) = π_k N(x|μ_k,Σ_k) / p(x) is the responsibility of component k.
    
    Args:
        x: Input samples [batch, dim]
        means: Component means [num_components, dim]
        covs: Component covariances [num_components, dim, dim]
        weights: Component weights (mixing coefficients) [num_components]
              Should sum to 1.
    
    Returns:
        score: ∇ log p(x) [batch, dim]
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Normalize weights to ensure they sum to 1
    weights = weights / weights.sum()
    
    # Compute log probabilities for each component
    # log N(x|μ_k,Σ_k) = -0.5 * [(x-μ_k)^T Σ_k^{-1} (x-μ_k) + log|Σ_k| + d*log(2π)]
    log_probs = torch.zeros(batch_size, num_components, device=x.device)
    component_scores = torch.zeros(batch_size, num_components, dim, device=x.device)
    
    for k in range(num_components):
        # Difference from mean
        diff = x - means[k]  # [batch, dim]
        
        # Compute inverse covariance and determinant
        cov_k = covs[k]  # [dim, dim]
        cov_inv = torch.inverse(cov_k)  # [dim, dim]
        cov_det = torch.det(cov_k)
        
        # Mahalanobis distance
        mahal = torch.sum(diff @ cov_inv * diff, dim=1)  # [batch]
        
        # Log probability
        log_probs[:, k] = -0.5 * (mahal + torch.log(cov_det) + dim * torch.log(torch.tensor(2 * torch.pi)))
        
        # Score for this component: Σ_k^{-1} (μ_k - x) = -Σ_k^{-1} (x - μ_k)
        component_scores[:, k] = -diff @ cov_inv  # [batch, dim]
    
    # Add log weights
    log_weights = torch.log(weights)  # [num_components]
    log_weighted_probs = log_probs + log_weights  # [batch, num_components]
    
    # Compute log p(x) using log-sum-exp trick for numerical stability
    log_p_x = torch.logsumexp(log_weighted_probs, dim=1, keepdim=True)  # [batch, 1]
    
    # Compute responsibilities: w_k(x) = π_k N(x|μ_k,Σ_k) / p(x)
    log_responsibilities = log_weighted_probs - log_p_x  # [batch, num_components]
    responsibilities = torch.exp(log_responsibilities)  # [batch, num_components]
    
    # Weighted sum of component scores
    # score = ∑_k w_k(x) * score_k(x)
    score = torch.sum(responsibilities.unsqueeze(2) * component_scores, dim=1)  # [batch, dim]
    
    return score


class SimpleFlowModel(nn.Module):
    """Simple MLP velocity field"""
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),  # +1 for time
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )
    
    def forward(self, x, t):
        """Predict velocity at (x, t)"""
        t_expand = t.view(-1, 1).expand(x.shape[0], 1)
        xt = torch.cat([x, t_expand], dim=1)
        return self.net(xt)

def compute_divergence_loss(model, x_t, t, x_0, x_1, mean_t, sigma_t, reg_weight=0.1):
    """
    Compute flow matching loss with divergence regularization
    
    L_CDM(θ) = E[ |(∇·u_t - ∇·v_t) + (u_t - v_t)·∇log p_t| ]
    
    Args:
        model: velocity field v_θ(x, t)
        x_t: interpolated samples [batch, dim]
        t: time [batch, 1]
        x_0: noise samples [batch, dim]
        x_1: data samples [batch, dim]
        sigma_t: noise schedule σ(t) [batch, 1]
        reg_weight: divergence regularization weight
    """
    x_t.requires_grad_(True)
    
    # 1. VELOCITY MATCHING LOSS (standard flow matching)
    # Target velocity: u_t(x|x1) = dx_t/dt = x_1 - x_0
    velocity_target = x_1 - x_0  # u_t
    
    # Predicted velocity: v_t(x, θ)
    velocity_pred = model(x_t, t)  # v_t
    
    # MSE loss
    velocity_loss = ((velocity_pred - velocity_target) ** 2).mean()
    
    
    # 2. CONDITIONAL DIVERGENCE MATCHING LOSS
    # Sample random vector ε ~ N(0, I) for Hutchinson's estimator
    epsilon = torch.randn_like(x_t)
    
    # TERM 2.a: (∇·u_t - ∇·v_t)
    # ---------------------------
    # NOTE: The following code is commented out because u_t = x_1 - x_0 is constant w.r.t. x_t
    # This means ∇·u_t = 0, so we can skip computing it and just use -∇·v_t
    # Original (broken) code:
    # u_dot_eps = (velocity_target * epsilon).sum()
    # div_u_grad = torch.autograd.grad(
    #     outputs=u_dot_eps,
    #     inputs=x_t,
    #     grad_outputs=torch.ones_like(u_dot_eps),
    #     create_graph=True,
    #     retain_graph=True
    # )[0]
    # div_u_t = (epsilon * div_u_grad).sum(dim=1)  # [batch]
    
    # Divergence of predicted velocity v_t using Hutchinson's estimator
    v_dot_eps = (velocity_pred * epsilon).sum()
    div_v_grad = torch.autograd.grad(
        outputs=v_dot_eps,
        inputs=x_t,
        grad_outputs=torch.ones_like(v_dot_eps),
        create_graph=True,
        retain_graph=True
    )[0]
    div_v_t = (epsilon * div_v_grad).sum(dim=1)  # [batch]
    
    # Difference of divergences (since div_u_t = 0)
    # divergence_diff = div_u_t - div_v_t  # Original
    divergence_diff = -div_v_t  # Simplified (since div_u_t = 0)
    
    # TERM 2.b: (u_t - v_t)·∇log p_t
    # ------------------------------
    # Score (single gauss): ∇ log p(x_t | x_1) = -(x_t - t*x_1) / σ²(t)
    score = -(x_t - mean_t) / (sigma_t ** 2)  # [batch, dim]
    
    # Velocity difference
    velocity_diff = velocity_target - velocity_pred  # [batch, dim]
    
    # Dot product with score (sum over dimensions for each batch)
    velocity_score_dot = (velocity_diff * score).sum(dim=1)  # [batch]
    
    
    # COMBINED DIVERGENCE LOSS
    # L_CDM = E[ |(∇·u - ∇·v) + (u - v)·score| ]
    cdm_loss = torch.abs(divergence_diff + velocity_score_dot).mean()
    
    
    # 3. TOTAL LOSS
    total_loss = velocity_loss + reg_weight * cdm_loss
    
    return total_loss, {
        'velocity_loss': velocity_loss.item(),
        'cdm_loss': cdm_loss.item(),
        'divergence_diff': divergence_diff.mean().item(),
        'velocity_score_dot': velocity_score_dot.mean().item(),
        'total_loss': total_loss.item()
    }


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Setup
    batch_size = 32
    dim = 2
    model = SimpleFlowModel(dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Sample a training batch
    x_1 = torch.randn(batch_size, dim)  # Data samples
    x_0 = torch.randn(batch_size, dim)  # Noise samples
    t = torch.rand(batch_size, 1)       # Random times
    
    # Interpolate
    x_t = t * x_1 + (1 - t) * x_0
    
    # Noise schedule (example: σ(t) = 1-t)
    sigma_t = 1 - t

    # Mean for score calculation
    mean_t = t * x_1
    
    # Compute loss
    loss, metrics = compute_divergence_loss(
        model, x_t, t, x_0, x_1, mean_t, 
        sigma_t, reg_weight=0.1
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Velocity Loss: {metrics['velocity_loss']:.4f}")
    print(f"CDM Loss: {metrics['cdm_loss']:.4f}")
    print(f"Total Loss: {metrics['total_loss']:.4f}")