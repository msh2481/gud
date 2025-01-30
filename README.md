# Unified Diffusion: Bridging Autoregressive and Diffusion Models

This project explores the effectiveness of different generative modeling approaches on synthetic time series data, with a particular focus on Unified Diffusion - a novel approach that combines the strengths of both autoregressive (AR) models and diffusion models.

## Theoretical Background

### The Challenge of Time Series Generation

Different types of time series exhibit distinct characteristics that make them more suitable for certain generative approaches:

- **Chaotic Time Series** (e.g., logistic map):
  - Strong local dependencies
  - Better handled by autoregressive models
  - Diffusion models may struggle unless using many time steps
  
- **Random Mean Series** (e.g., white noise with random mean):
  - Global structure with local randomness
  - Excellently handled by diffusion models
  - Autoregressive models may show divergent behavior unless perfectly fitted

### Unified Diffusion Approach

We introduce Unified Diffusion, a flexible framework that can adapt to different types of time series by varying its denoising schedule:

1. **Token-Specific Noise Schedules**: Each token in the sequence has an independent noise schedule

2. **Configurable Denoising Patterns**:
   - **Pure Autoregressive**: Token i+1 starts denoising only after token i is completely denoised
   - **Pure Diffusion**: All tokens denoise simultaneously
   - **Hybrid Approach**: Tokens denoise from left to right with overlapping denoising intervals

## Synthetic Data Generation

The project includes synthetic data generation capabilities (`main.py`) that create time series combining:
- Chaotic dynamics from the logistic map
- White noise with random mean

The mixture is controlled by a `chaos_ratio` parameter:
```python
data = chaos_ratio logistic_map + (1 - chaos_ratio) random_noise
```

### Research Hypothesis

Our key hypothesis is that for certain combinations of chaotic and random components, the optimal model will be neither purely autoregressive nor purely diffusive, but rather a hybrid approach with partially overlapping denoising schedules.

This synthetic dataset serves as a controlled environment to:
1. Validate the theoretical advantages of Unified Diffusion
2. Demonstrate scenarios where hybrid approaches outperform pure AR or diffusion models
3. Study the relationship between data characteristics and optimal denoising schedules

## Model Architecture

The Unified Diffusion model extends the DDPM (Denoising Diffusion Probabilistic Models) framework by introducing position-specific noise schedules. The core architecture maintains the denoising score matching objective but allows for independent noise levels per token.

### Training Objective

The model is trained to predict the noise component given a partially noised input:

```python
eps_theta(sqrt(gamma) * x_0 + sqrt(1 - gamma) * eps, gamma) -> eps
```

where:
- `eps_theta` is the model
- `gamma` is a vector of current signal-to-noise ratios (γ = ∏ α_t = ∏ (1 - β_t))
- `eps` is standard normal distributed noise
- `x_0` is the original clean data

### Sampling Process

Sampling follows Langevin dynamics with position-specific noise schedules:

```python
x[t-1] = 1/sqrt(alpha[t]) * (x[t] - beta[t]/sqrt(1 - gamma[t]) * eps_theta) + sigma * randn()
```

Key parameters:
- `alpha[t] = gamma[t] / gamma[t-1]` (noise schedule step)
- `beta[t] = 1 - alpha[t]` (noise level)
- `sigma[t] = beta[t]` (sampling noise, following DDPM)

This formulation allows for flexible denoising patterns while maintaining the theoretical guarantees of diffusion models.

## Getting Started

To generate synthetic data:
```python
python main.py
```


This will create sample time series and display them using matplotlib.

## Project Structure

- `main.py`: Synthetic data generation utilities
  - `gen()`: Generates single time series
  - `gen_dataset()`: Creates batches of time series