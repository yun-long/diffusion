# Flow and Diffusion

![diffusion](docs/representative.gif)


Both **diffusion models** and **flow matching** share the same overarching **goal**: to learn a model that captures the data distribution. In other words, they aim to maximize the likelihood of the observed data under the model:

```math
\text{Goal:}\quad \max_\theta \; \log p_\theta(x_0).
```

Since directly optimizing the likelihood is intractable, both approaches transform it into a **tractable regression problem**. By introducing a time-dependent process (noise perturbations in diffusion, velocity fields in flow matching), the objective becomes:

```math
\text{Optimization objective:}\quad 
\min_\theta \; 
\mathbb{E}_{t, x_0, y}
\Big[
  \|\, y - f_\theta(x_t, t)\,\|^2
\Big],
```

where

* $x_t$ is the data perturbed by the chosen time-dependent process,
* $y$ is the regression target (e.g., noise in diffusion, velocity in flow matching),
* $f_\theta$ is the model being trained.

This general formulation highlights the **common principle**: both methods convert maximum-likelihood learning into a supervised regression task, making the optimization stable and effective.

### Fundamentals 

Perfect 🚀 Here’s a structured **roadmap / tutorial outline** you can directly use in your README, blog, or lecture notes. It breaks the learning path into clear levels with progressively deeper concepts.


--- 
### Foundations: Math & Intuition

1. **Probability Basics**

   * Random variables, probability distributions, expectations
   * Log-likelihood and maximum likelihood estimation (MLE)

2. **Dynamics over Time**

   * Ordinary Differential Equations (ODEs): deterministic dynamics
   * Stochastic Differential Equations (SDEs): adding randomness to dynamics

3. **Supervised Learning & Regression**

   * Squared error loss as a regression objective
   * Connection to modeling simple transformations of data

### Intermediate: Core Generative Modeling Tools

1. **Information-Theoretic Foundations**

   * KL divergence: measuring distribution mismatch
   * Jensen’s inequality: the basis of variational bounds
   * ELBO (Evidence Lower Bound): making maximum likelihood tractable

2. **Diffusion Models**

   * Forward process: progressively adding noise to data
   * Reverse process: denoising with a learned model
   * Noise prediction objective (regression on noise)

3. **Flow Matching**

   * Velocity fields and continuity equations
   * Matching flows to recover data distributions
   * Regression on target velocities instead of noise


### Advanced: Limitations & Frontiers

1. **Limitations of Standard Approaches**

   * Many-step sampling → slow inference
   * Dependence on numerical solvers

2. **Accelerating Generative Models**

   * One-step / few-step prediction
   * Distillation and model compression for faster sampling

3. **Consistency Models**

   * Predicting consistently across time scales
   * Bridging iterative and direct sampling


--- 
### References

**Course**

- [Introduction to Flow Matching and Diffusion Models (MIT, Peter Holderrieth and Ezra Erives)](https://diffusion.csail.mit.edu/) 
- [Deep Generative Models (MIT, Prof. Kaiming He)](https://mit-6s978.github.io/schedule.html)


**Blog**

- [An Introduction to Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)