# Bernstein-Dispersion-Relations
These Python scripts evaluate the roots of electron and ion Bernstein waves in a multi-species plasma in the perpendicular and electrostatic limit.
Three different methods are provided: A root-finding method based on Newton-Raphon's method; a direct evaluation from a closed form of $k(\omega)$ using an estimative expression, suitable for the fundamental dispersion branch; an eigenvalue problem formulation.

**The dispersion relation**\
```\mathcal{D}(k,\omega) = k^2 + 2\sum_\xi \frac{\omega_{p\xi}^2}{v_{th,\xi}^2}\sum_{n=-\infty}^\infty \frac{n}{n-\alpha_{\xi}}\exp(-\lambda_\xi)I_n(\lambda_\xi)
```
