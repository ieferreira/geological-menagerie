import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import re
import seaborn as sns


# Loading annex 1 and 2

ann1 = pd.read_excel("2021GarciaAnexo1.xlsx")
ann2 = pd.read_excel("2021GarciaAnexo2.xlsx")


ann1.head()


ann2.head()


from pyrolite.plot import pyroplot


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






