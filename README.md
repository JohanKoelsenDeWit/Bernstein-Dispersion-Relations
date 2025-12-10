# Bernstein-Dispersion-Relations
These Python scripts evaluate the roots of electron and ion Bernstein waves in a multi-species plasma in the perpendicular and electrostatic limit.
Three different methods are provided: A root-finding method based on Newton-Raphon's method; a direct evaluation from a closed form of $k(\omega)$ using an estimative expression, suitable for the fundamental dispersion branch; an eigenvalue problem formulation.


All equations are written in the paper "An introduction to perpendicularly propagating Bernstein waves".

Root-finding.py uses the root-finding scheme to compute the f(k) dispersion branches.
Expansion-method.py uses the estimative dispersion relation.
EVP-method.py uses the eigenvalue problem formulation.

Lastly, Combined-methods.py compares the three methods for the fundamental electron Bernstein branch:
<img width="2720" height="1320" alt="CombinedMethods" src="https://github.com/user-attachments/assets/de19068c-4801-4ef9-a2e9-63ba337b99fe" />

