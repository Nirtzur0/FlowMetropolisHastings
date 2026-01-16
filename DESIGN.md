# DiffMCMC Design Document

## 1. Overview
DiffMCMC is a library for exact Bayesian inference that uses a learned Continuous Normalizing Flow (CNF) as a global proposal distribution within a Metropolis-Hastings (MH) sampler. The key idea is to "accelerate" mixing by proposing large jumps across the posterior using the learned model, while maintaining exactness (correct stationary distribution) by rejecting samples that don't satisfy the MH criterion.

## 2. Mathematical Foundation

### 2.1 Mixture Kernel MH
The transition kernel is a mixture:
$$ T(x'|x) = (1 - p_{global}) K_{local}(x'|x) + p_{global} K_{global}(x'|x) $$

- **Local Kernel $K_{local}$**: Standard RWM or MALA. Ensures ergodicity and exploring local details.
- **Global Kernel $K_{global}$**: An independence sampler proposal $q_\phi(x')$.
    - Note: $K_{global}(x'|x) = q_\phi(x')$. The proposal does not depend on current state $x$.

### 2.2 Acceptance Ratio (Global Move)
For an independence sampler proposal $x' \sim q(x')$, the MH ratio is:
$$ \alpha = \min\left(1, \frac{\pi(x') q(x)}{\pi(x) q(x')} \right) $$

**Requirement**: We must be able to evaluate $\log q(x)$ and $\log q(x')$.

### 2.3 Flow Matching / Rectified Flow
We model $q_\phi$ using a Vector Field $v_\phi(x, t)$.
- **Generative Process** ($z \to x$):
  $$ dx_t = v_\phi(x_t, t) dt, \quad x_0 \sim N(0, I) $$
  $$ x_{sampled} = x_1 $$

- **Density Evaluation** ($x \to z$):
  By the instantaneous change of variables formula:
  $$ \log q_1(x_1) = \log q_0(x_0) - \int_0^1 \text{div}(v_\phi(x_t, t)) dt $$
  where $x_0$ is obtaining by solving the ODE backwards from $x_1$.

- **Hutchinson Trace Estimator**:
  $$ \text{div}(v) = \text{Tr}\left(\frac{\partial v}{\partial x}\right) \approx \epsilon^T \left(\frac{\partial v}{\partial x} \epsilon\right), \quad \epsilon \sim N(0, 1) $$
  To ensure reversibility for the MH step, we must either:
  1. Use exact divergence (expensive for high dim).
  2. Use the same "randomness" for evaluating $q(x)$ every time $x$ is visited. This implies hashing $x$ to seed $\epsilon$ or storing the computed $\log q(x)$ with the chain state. We will store $\log q(x)$ in the chain state so re-evaluation isn't needed for the *current* point, only the *proposed* point.

### 2.4 Inference Modes
DiffMCMC exposes explicit inference modes with contracts:
- **exact**: delayed-acceptance with an exact proposal density; requires deterministic log-density evaluation and a consistent integration grid. Warnings are emitted when conditions are not met.
- **pseudo_marginal**: requires an unbiased estimator of $q(x)$ in an extended state space.
- **approx**: uses approximate $q(x)$ for speed; may introduce bias.

## 3. Architecture

### 3.1 Project Structure
```
diffmcmc/
  core/
    mh.py           # The Sampler class
    kernels.py      # AbstractKernel, RWMKernel, MALAKernel
  proposal/
    flow.py         # ContinuousNormalizingFlow class
    nets.py         # MLP time-dependent networks
    train.py        # Online training logic
  targets/          # Toy distributions
  tests/            # Unittests
```

### 3.2 Key Classes

#### `DiffusionMH`
- **State**: `current_x`, `current_log_prob`, `current_log_q` (if available).
- **Methods**:
    - `step()`: Performs one MCMC step.
    - `warmup()`: Runs local sampler to fill buffer.
    - `train()`: Triggers flow matching training.

#### `ContinuousNormalizingFlow`
- **Wraps**: A `torch.nn.Module` velocity network.
- **Methods**:
    - `sample(n)`: Returns batch of samples.
    - `log_prob(x)`: Returns log density.

## 4. Trade-offs & Decisions

- **ODE Solver**: Use `torchdiffeq` or implementing a simple fixed-step RK4 for standard PyTorch usage?
  - *Decision*: Implement a simple RK4 / Euler first to keep dependencies minimal (only torch). It's faster for training/sampling usually than full adaptive solvers if the flow is straight (Rectified Flow).
- **Training Frequency**:
  - *Decision*: "Safe Mode" MVP. Train once after warmup, then freeze. Adaptive retraining is a stretch.
- **Density Estimation**:
  - *Decision*: Store `log_q` of the current state to avoid re-computation. When proposing $x'$, compute $\log q(x')$. If accepted, cache it.
- **Integrator Contract**:
  - *Decision*: Enforce step-size divisibility for stable grids; optionally compute discrete-step logdet for exactness on small dimensions.

## 5. Technology Stack
- **Python 3.11+**
- **PyTorch**: For NN and autodiff.
- **NumPy/SciPy**: For interfacing with generic log_probs.
