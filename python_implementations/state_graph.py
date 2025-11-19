import torch
import numpy as np

def construct_state_graph(x):
    """
    Construct a separate graph for each time step, showing how all batch samples
    are connected at that specific time.
    
    Args:
        x: Interpolated samples [batch, n_steps, dim]
    
    Returns:
        graphs: List of dictionaries, one per time step, each containing:
            - 'time_step': int
            - 'nodes': tensor [batch, dim] - states at this time step
            - 'distances': tensor [batch, batch] - pairwise distances
            - 'adjacency_list': dict mapping node_idx -> [(neighbor_idx, distance), ...]
    """
    batch_size, n_steps, dim = x.shape
    
    graphs = []
    
    for step in range(n_steps):
        # Extract all states at this time step [batch, dim]
        nodes_at_step = x[:, step, :]  # [batch, dim]
        
        # Compute pairwise distances between all states at this time step
        # Using broadcasting: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
        nodes_norm_sq = (nodes_at_step ** 2).sum(dim=1, keepdim=True)  # [batch, 1]
        distances = torch.sqrt(
            nodes_norm_sq + nodes_norm_sq.T - 2 * nodes_at_step @ nodes_at_step.T
        )  # [batch, batch]
        
        # Create adjacency list for this time step
        adjacency_list = {}
        for i in range(batch_size):
            neighbors = []
            for j in range(batch_size):
                if i != j:  # Don't include self-loops
                    dist = distances[i, j].item()
                    neighbors.append((j, dist))
            adjacency_list[i] = neighbors
        
        graphs.append({
            'time_step': step,
            'nodes': nodes_at_step,
            'distances': distances,
            'adjacency_list': adjacency_list
        })
    
    return graphs

def compute_divergence_numerical(graphs, dx, k=4):
    """
    Compute divergence of the flow field at each state using numerical differentiation
    based on k-nearest neighbors in the state graph.
    
    Args:
        graphs: List of state graphs (one per time step) from construct_state_graph
        dx: Flow field [batch, n_steps, dim] - velocity at each state
        k: Number of nearest neighbors to use for numerical differentiation
    
    Returns:
        divergence: tensor [batch, n_steps] - divergence at each state
    """
    batch_size = dx.shape[0]
    n_steps = len(graphs)
    dim = dx.shape[2]
    
    divergence = torch.zeros(batch_size, n_steps)
    
    for step, graph in enumerate(graphs):
        nodes = graph['nodes']  # [batch, dim]
        adjacency_list = graph['adjacency_list']
        
        for i in range(batch_size):
            # Get k-nearest neighbors
            neighbors = sorted(adjacency_list[i], key=lambda x: x[1])[:k]
            
            if len(neighbors) == 0:
                continue
            
            # Current state and velocity
            x_i = nodes[i]  # [dim]
            v_i = dx[i, step]  # [dim]
            
            # Estimate divergence using finite differences
            # ∇·v ≈ Σ_d (∂v_d/∂x_d)
            div_estimate = 0.0
            
            for d in range(dim):
                # For each dimension, use neighbors to estimate ∂v_d/∂x_d
                numerator = 0.0
                denominator = 0.0
                
                for neighbor_idx, dist in neighbors:
                    x_j = nodes[neighbor_idx]  # [dim]
                    v_j = dx[neighbor_idx, step]  # [dim]
                    
                    # Directional difference in dimension d
                    dx_d = x_j[d] - x_i[d]
                    dv_d = v_j[d] - v_i[d]
                    
                    # Weight by inverse distance (avoid division by zero)
                    if abs(dx_d.item()) > 1e-8:
                        weight = 1.0 / (dist + 1e-8)
                        numerator += weight * (dv_d / dx_d)
                        denominator += weight
                
                if denominator > 0:
                    div_estimate += (numerator / denominator).item()
            
            divergence[i, step] = div_estimate
    
    return divergence

def compute_divergence_numerical_lsq(graphs, dx, k=4):
    """
    Compute divergence using least-squares fitting of the velocity field
    based on k-nearest neighbors. More robust than simple finite differences.
    
    For each state x_i, we fit a local linear model:
    v(x) ≈ v_i + J·(x - x_i)
    
    where J is the Jacobian. Then ∇·v = trace(J).
    
    Args:
        graphs: List of state graphs (one per time step)
        dx: Flow field [batch, n_steps, dim]
        k: Number of nearest neighbors to use
    
    Returns:
        divergence: tensor [batch, n_steps]
    """
    batch_size = dx.shape[0]
    n_steps = len(graphs)
    dim = dx.shape[2]
    
    divergence = torch.zeros(batch_size, n_steps)
    
    for step, graph in enumerate(graphs):
        nodes = graph['nodes']  # [batch, dim]
        adjacency_list = graph['adjacency_list']
        
        for i in range(batch_size):
            # Get k-nearest neighbors
            neighbors = sorted(adjacency_list[i], key=lambda x: x[1])[:k]
            
            if len(neighbors) < dim:  # Need at least dim neighbors for LSQ
                continue
            
            # Current state and velocity
            x_i = nodes[i]  # [dim]
            v_i = dx[i, step]  # [dim]
            
            # Build matrices for least-squares
            # A: [num_neighbors, dim] - position differences
            # B: [num_neighbors, dim] - velocity differences
            A = []
            B = []
            
            for neighbor_idx, dist in neighbors:
                x_j = nodes[neighbor_idx]
                v_j = dx[neighbor_idx, step]
                
                A.append((x_j - x_i).cpu().numpy())
                B.append((v_j - v_i).cpu().numpy())
            
            A = np.array(A)  # [k, dim]
            B = np.array(B)  # [k, dim]
            
            # Solve for Jacobian: J^T ≈ (A^T A)^-1 A^T B^T
            # We want J such that B ≈ A @ J^T
            # So J^T ≈ lstsq(A, B)
            try:
                J_T = np.linalg.lstsq(A, B, rcond=None)[0]  # [dim, dim]
                J = J_T.T  # [dim, dim]
                
                # Divergence = trace(J)
                div_value = float(np.trace(J))
                divergence[i, step] = div_value
            except np.linalg.LinAlgError:
                # If singular, use simple averaging
                divergence[i, step] = 0.0
    
    return divergence

def plot_paths(x, t):
    """
    Visualize the paths taken by samples over time.
    
    Args:
        x: Interpolated samples [batch, n_steps, dim]
        t: Time steps [batch, n_steps, 1]
    """
    import matplotlib.pyplot as plt
    
    batch_size, n_steps, dim = x.shape
    plt.figure(figsize=(8, 8))
    
    for i in range(batch_size):
        path = x[i].cpu().numpy()  # [n_steps, dim]
        plt.plot(path[:, 0], path[:, 1], marker='o')
    
    plt.title("Sample Paths Over Time")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Setup
    batch_size = 64
    dim = 2
    
    # Sample a training batch
    x_1 = torch.ones(batch_size, dim)  # Data samples [batch, dim]
    x_0 = torch.randn(batch_size, dim)  # Noise samples [batch, dim]
    print("x_1 shape:", x_1.shape)  # Should be [batch, dim]
    print("x_0 shape:", x_0.shape)  # Should be [batch

    # uniformly sample times [batch, n_steps, 1]
    n_steps = 10
    t = torch.linspace(0, 1, n_steps).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)  # [batch, n_steps, 1]
    print("t shape:", t.shape)  # Should be [batch, n_steps, 1]

    # Interpolate with linear schedule [batch, n_steps, dim]
    x = t * x_1.unsqueeze(1) + (1 - t) * x_0.unsqueeze(1)
    print("x shape:", x.shape)  # Should be [batch, n_steps, dim]

    # create the flow field samples at each time step
    # v(x_t) = x_1 - x_t (pointing toward goal from current position)
    dx = x_1.unsqueeze(1) - x  # [batch, n_steps, dim]
    
    # Construct state graph
    print("\n=== Constructing State Graphs (one per time step) ===")
    graphs = construct_state_graph(x)
    print(f"Number of time steps: {len(graphs)}")
    print(f"Nodes per time step: {graphs[0]['nodes'].shape[0]}")
    print(f"Distance matrix shape per time step: {graphs[0]['distances'].shape}")
    
    # Analyze first time step (t=0)
    print("\n=== Time Step 0 ===")
    graph_t0 = graphs[0]
    distances_t0 = graph_t0['distances']
    non_diag_t0 = distances_t0[~torch.eye(distances_t0.shape[0], dtype=bool)]
    print(f"Min distance: {non_diag_t0.min():.4f}")
    print(f"Max distance: {non_diag_t0.max():.4f}")
    print(f"Mean distance: {non_diag_t0.mean():.4f}")
    
    # Find k-nearest neighbors at time step 0
    k = 5
    node_idx = 0
    neighbors_t0 = sorted(graph_t0['adjacency_list'][node_idx], key=lambda x: x[1])[:k]
    print(f"\n{k}-Nearest neighbors of node {node_idx} at time step 0:")
    for neighbor_idx, dist in neighbors_t0:
        print(f"  Node {neighbor_idx}: distance = {dist:.4f}")
    
    # Analyze last time step
    print(f"\n=== Time Step {n_steps-1} ===")
    graph_tf = graphs[-1]
    distances_tf = graph_tf['distances']
    non_diag_tf = distances_tf[~torch.eye(distances_tf.shape[0], dtype=bool)]
    print(f"Min distance: {non_diag_tf.min():.4f}")
    print(f"Max distance: {non_diag_tf.max():.4f}")
    print(f"Mean distance: {non_diag_tf.mean():.4f}")
    
    # Compute divergence numerically using nearest neighbors
    print("\n=== Computing Numerical Divergence ===")
    k_neighbors = 4
    
    # Method 1: Weighted finite differences
    div_fd = compute_divergence_numerical(graphs, dx, k=k_neighbors)
    print(f"Divergence (finite diff) shape: {div_fd.shape}")
    print(f"Mean divergence: {div_fd.mean():.6f}")
    print(f"Std divergence: {div_fd.std():.6f}")
    print(f"Divergence at batch 0, step 0: {div_fd[0, 0]:.6f}")
    print(f"Divergence at batch 0, step {n_steps-1}: {div_fd[0, -1]:.6f}")
    
    # Method 2: Least-squares Jacobian fitting (more robust)
    div_lsq = compute_divergence_numerical_lsq(graphs, dx, k=k_neighbors)
    print(f"\nDivergence (LSQ) shape: {div_lsq.shape}")
    print(f"Mean divergence: {div_lsq.mean():.6f}")
    print(f"Std divergence: {div_lsq.std():.6f}")
    for i in range(n_steps):
        print(f"Divergence at batch 0, step {i}: {div_lsq[0, i]:.6f}")
        print(f"Divergence at batch 10, step {i}: {div_lsq[10, i]:.6f}")
        print(f"Divergence at batch 20, step {i}: {div_lsq[10, i]:.6f}")
        print(f"Divergence at batch 30, step {i}: {div_lsq[10, i]:.6f}")



