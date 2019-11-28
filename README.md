# Kernel_conditional_density_operators
Code accompanying the paper "Kernel conditional density operators"

The main interface for CDOs is the "CorrectedConditionDensityOperator" class in rkhsop/op/cond_probability.py

## Optimized code for LSCDE in Conditional_Density_Estimation/

In order to do experiments in much quicker time and without Monte Carlo variance,
we modified the LSCDE code found at https://github.com/freelunchtheorem/Conditional_Density_Estimation
to compute mean and variance of an LSCDE in closed form rather than from samples.
A quick sanity check shows that Monte Carlo estimates of these two statistical moments indeed
converge to our closed form estimates.