# General imports
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import re
import seaborn as sns

# Pyrolite Pyroplot
from pyrolite.plot import pyroplot

# Pyrolite SVC pipeline
from pyrolite.util.skl.pipeline import SVC_pipeline
from pyrolite.util.skl.vis import plot_mapping
from pyrolite.util.plot import DEFAULT_DISC_COLORMAP



def clean_df(df):
    clean_df = df.copy()
    clean_df = clean_df.apply(lambda x: x.astype(str).str.replace(",", "."))
    clean_df = clean_df.apply(lambda x: x.astype(str).str.replace("<", "")) # Remove less than with empty space, :/
    clean_df = clean_df.replace('-', None)
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce')
    return clean_df


# Loading annex 1 and 2

ann1 = pd.read_excel("2021GarciaAnexo1.xlsx")
ann2 = pd.read_excel("2021GarciaAnexo2.xlsx")


ann1.head()


ann2.head()


oxides = ["SiO2", "MgO", "CaO"]
ann1Ternary = ann1[oxides].copy()
ann1Ternary = ann1Ternary.apply(lambda x: x.str.replace(",", "."))
ann1Ternary = ann1Ternary.apply(lambda x: x.str.replace("<", "")) # Remove less than with empty space, :/
ann1Ternary["Class"] = ann1["Código"]
ann1Ternary.Class = ann1Ternary.Class.apply(lambda x: x[:2]) # Leave only first 2 letters of code to make three classes



ann1Ternary.head()


ann1Ternary[oxides].astype('float').describe()


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 9))

ann1Ternary[oxides][ann1Ternary["Class"]=="It"].pyroplot.scatter(ax=ax[0], c="k").set_title("Río Itoco\n\n\n")
ann1Ternary[oxides][ann1Ternary["Class"]=="Pa"].pyroplot.scatter(ax=ax[1], c="g").set_title("Quebrada La Pava\n\n\n")
ann1Ternary[oxides][ann1Ternary["Class"]=="Tu"].pyroplot.scatter(ax=ax[2], c="r").set_title("Túneles\n\n\n")


ax = fig.orderedaxes  # creating scatter plots reorders axes, this is the correct order
plt.tight_layout()


svc_ann1 = clean_df(ann1[['SiO2', 'Al2O3', 'TiO2', 'Fe2O3T', 'MgO', 'CaO', 'Na2O','K2O', 'P2O5']]).to_numpy()


target = ann1.copy()["Código"]
target = target.apply(lambda x: x[:2])




mapping = {'It': 0, 'Pa': 1, 'Tu':2}
target.replace(mapping, inplace=True)


type(svc_ann1[0][0])


svc = SVC_pipeline(probability=True)
gs = svc.fit(svc_ann1, target)


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

a, tfm, mapped = plot_mapping(
    svc_ann1, gs.best_estimator_, ax=ax[1], s=50, init="pca"
)
ax[0].scatter(*mapped.T, c=DEFAULT_DISC_COLORMAP(gs.predict(svc_ann1)), s=50)

ax[0].set_title("Predicted Classes")
ax[1].set_title("With Relative Certainty")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])



