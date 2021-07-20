# Sandstone data clasification

Classifying sandstone csv data.

Classes in .csv file are

- monocrystalline quartz (Qm)
- undulose monocrystalline quartz (Qmu)
- polycrystalline quartz (Qp)
- plagioclase feldspar (Plag)
- alkali feldspar (Afsp)
- lithic fragments (Lf)
- cement/pseudo matrix (Cem+PM)
- pore space (Pore)

I use first a K-Means Clustering algorithm and then a neural network classifier to distinguish between given rocks given its petrographical composition (it will reproduce Pettijohn classification triangle in the best case).

# References

- Dataset obtained from Bassis et al (2016), and csv obtained from https://github.com/trqmorgan/QFL-Pettijohn-Ternary
