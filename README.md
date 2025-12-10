# Bernstein-Dispersion-Relations
These Python scripts evaluate the roots of electron and ion Bernstein waves in a multi-species plasma in the perpendicular and electrostatic limit.
Three different methods are provided: A root-finding method based on Newton-Raphon's method; a direct evaluation from a closed form of $k(\omega)$ using an estimative expression, suitable for the fundamental dispersion branch; an eigenvalue problem formulation.


All equations are written in the paper "An introduction to perpendicularly propagating Bernstein waves".

Root-finding.py uses the root-finding scheme to compute the f(k) dispersion branches.
Expansion-method.py uses the estimative dispersion relation.
EVP-method.py uses the eigenvalue problem formulation.
