# Unified Diffusion: Bridging Autoregressive and Diffusion Models

This project explores the effectiveness of different generative modeling approaches on synthetic time series data, with a particular focus on Unified Diffusion - a novel approach that combines the strengths of both autoregressive (AR) models and diffusion models.

## Theoretical Background

### Unified Diffusion Approach

We introduce Unified Diffusion, a flexible framework that can adapt to different types of time series by varying its denoising schedule:

1. **Token-Specific Noise Schedules**: Each token in the sequence has an independent noise schedule

2. **Configurable Denoising Patterns**:
   - **Pure Autoregressive**: Token i+1 starts denoising only after token i is completely denoised
   - **Pure Diffusion**: All tokens denoise simultaneously
   - **Hybrid Approach**: Tokens denoise from left to right with overlapping denoising intervals

## Model Architecture

$$ 
\begin{split}
\alpha(p, t) &= \frac{\alpha(t)}{\alpha(p)} \\
 \beta(p, t) &= 1 - \alpha(p, t) =  \\
 s(t) &= \frac{\alpha(t)}{\beta(t)} \\
q(x(t) \mid x(p)) &= \mathcal{N}(\sqrt{\alpha(p, t)} x(p),\ \beta(p, t)) \\
q(x(p) \mid x(t), x(0)) &= \mathcal{N}(\hat{\mu}, \hat{\sigma}^2), \quad \text{where}\\
 \hat{\mu} &= \frac{\beta(p, t)}{\beta(t)} \cdot \sqrt{\alpha(0, p)} x(0) + \frac{\beta(p)\alpha(p, t)}{\beta(t)} \cdot \sqrt{\alpha(t, p)} x(t) \\
 \hat{\sigma}^2 &= \alpha(t, p) ( \beta(p, t) \parallel \beta(0, p)\alpha(p, t)) = \frac{\beta(p)}{\beta(t)} \beta(p, t) \\
 p(x(p) \mid x(t)) &= q(x(p) \mid x(t), \hat{x}(0)) \\
 \Delta \hat{\mu} &= \frac{\beta(p, t)}{\beta(t)} \sqrt{\alpha(p)} \cdot \Delta \hat{x}(0) \\
 \Delta \hat{x}(0) &= \frac{1}{\sqrt{s(t)}} \cdot \Delta \hat{\varepsilon} \\
 KL(q \parallel p) &= \frac{1}{2} \frac{\Delta \hat{\mu}^2}{\hat{\sigma}^2} = \frac{1}{2} \frac{\beta(p, t) \alpha(p)}{\beta(p) \beta(t)} \cdot \Delta \hat{x}(0)^2 = \\
 &= \frac{1}{2}(s(p) - s(t)) \cdot \Delta \hat{x}(0)^2 \\
 &= \frac{1}{2}\frac{s(p) - s(t)}{s(t)} \cdot \Delta \hat{\varepsilon}^2 \\
 &\approx \frac{1}{2}(\log s(p) - \log s(t)) \cdot \Delta \hat{\varepsilon}^2 \\
\end{split}
$$

So the continuous ELBO is:
$$ 
\begin{split}
\mathcal{L} &= \frac{1}{2} \int_0^1 \left(\text{MSE in $x(0)$ space}\right) s'(t) d t \\
&= \frac{1}{2} \int_0^1 \left(\text{MSE in $\varepsilon$ space}\right) (\log s(t))' d t \\
&= \frac{1}{2} \int_0^1 \frac{\left(\text{MSE in $\varepsilon$ space}\right)}{s(t)} s'(t) d t \\
\end{split}
$$


For $L \leq s(t) \leq R$ region (with $L, R \gg 1$) we can upper bound the loss by $\mathcal{O}(\log \frac{R}{L})$ by predicting $\varepsilon = 0$.

For $0 \leq s(t) \leq r$ region (with $r \ll 1$) we can upper bound the loss by $\mathcal{O}(r)$ by predicting $x(0) = 0$.

Such baseline also means that it makes sense to sample SNR logarithmically for large values and linearly for small.

## Noise schedule

$$
\begin{split}
l(0) &= 0 \\
r(N - 1) &= 1 \\
r(i) - l(i) &= t \\
v &= \frac{N - 1}{1 - t}\\
w &= v \cdot t = \frac{t}{1 - t} (N - 1) \\
t &= \frac{w}{N - 1 + w} \\
l(i) &= \frac{i}{v} \quad r(i) = l(i) + t\\
\end{split}
$$


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

