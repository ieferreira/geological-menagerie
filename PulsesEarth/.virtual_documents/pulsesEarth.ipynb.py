import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import amp_spec as spec


df = pd.read_excel("geological_events.xlsx")
df.columns


df # original dataframe as is in the paper


# replace non numeric values with Nan
df.replace("—", np.nan, inplace=True)
df.replace("____", np.nan, inplace=True)
df
orig_df = df.copy()



dates_df = pd.DataFrame()
dates_df[["Start", "End"]] = orig_df["IntervalMa"].str.split("–", expand=True)


dates_df["Start"] = pd.to_numeric(dates_df["Start"])
dates_df["End"] = pd.to_numeric(dates_df["End"])

dates_df['Mid'] = dates_df.mean(axis=1)


orig_df


marine_extinction = df["MarineExtinction"]
anoxic_event = df["AnoxicEvent"]
continental_basalt = df["ContinentalBasalt"]
sequence_boundary = df["SequenceBoundary"]
nonmarine_extinction = df["NonMarineExtinction"]
changes_spreading_rate = df["ChangesSpreadingRate"]
intraplate_volcanism = df["IntraPlateVolcanism"]

data_columns = [marine_extinction, anoxic_event, continental_basalt, sequence_boundary, nonmarine_extinction, changes_spreading_rate, intraplate_volcanism]


data_cols = data_columns.copy()


def cleanColumns(column: pd.Series) -> list:
    
    ls = list(column)
    ls_temp = [] 
    for i in ls:
        ls_temp += str(i).split(";")

    ls_temp = [i.split("±") for i in ls_temp]
    


    def extract(lst):
        return [item[0] for item in lst if type(item)==list]

    def convertFloat(lst):
        return [float(item) for item in lst]

    result = extract(ls_temp)
    return list(result)


temp_dates = []
for i in data_columns:
    i = cleanColumns(i)
    
    temp_dates.append(i)




raw_dates = []
for i in temp_dates:
    i = [x.split(",") for x in i]
    raw_dates.append(i)





def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def convertFloat(lst):
    return [float(item) for item in lst]


# flatten_twice
list_dates = flatten_list(flatten_list(raw_dates))
# convert all elements to float
float_dates = convertFloat(list_dates)
# clean of Nans
cleaned_dates = [x for x in float_dates if str(x) get_ipython().getoutput("= 'nan']")


len(cleaned_dates) # total of events is correct!


max(cleaned_dates)


sns.histplot(cleaned_dates, bins=89)


df.head()


df_num = df.copy()[["MarineExtinction",	"AnoxicEvent",	"ContinentalBasalt",	"SequenceBoundary",	"NonMarineExtinction",	"ChangesSpreadingRate",	"IntraPlateVolcanism"]]


# Create a dictionary of the cleaned events to make a dataframe 

dict_events = {}
for i in data_columns:
    col = cleanColumns(i)
    dict_events[i.name] = col

cleaned_dict = {}
for k, v in dict_events.items():
    v = [i.split(",") for i in v]
    v = flatten_list(v)
    v = [float(j) for j in v]
    cleaned_dict[k] = v
    
names = []
for i in data_cols:
    names.append(i.name)



# Make a dataframe of events
df = pd.DataFrame(list(cleaned_dict.items()), columns=["Event", "Ages"]) 
dfT = df.T
dfT.columns= names
df_events = dfT.iloc[1: , :]


# Put each date on a row
iterdf = pd.DataFrame()
for i in ["MarineExtinction",	"AnoxicEvent",	"ContinentalBasalt",	"SequenceBoundary",	"NonMarineExtinction",	"ChangesSpreadingRate",	"IntraPlateVolcanism"]:
    col = df_events.explode(i).reset_index(drop=True)[i]
    iterdf[i] = col


iterdf.head()


# Count rows with events to group them in area plot
count_events = iterdf.notnull().astype('int')

# pick mid year as average of interval age (mya) given in paper
count_events["Mid"] = dates_df["Mid"]
count_events.rename(columns={"Mid": "Mya"}, inplace=True)
count_events.head()


count_events.plot.area(x="Mya", title="Cummulative number of geological events", figsize=(15,6))


dates = pd.DataFrame(cleaned_dates)
dates.columns = ["events"]



dates_list = dates["events"].tolist()
dates_list = sorted(dates_list)



resample = np.linspace(0,260,260)
df_resampled = dates.reindex(dates.index.union(resample)).loc[resample]
df_resampled = df_resampled.reset_index()
df_resampled = df_resampled.round(1)
df_resampled.drop(columns=["events"], inplace=True)
df_resampled["events"] = 0
df_resampled.rename(columns={"index": "rate"}, inplace=True)


cnt = 0
for index, row in df_resampled.iterrows():
    
    for i in list_dates:
        
            
        if(0 <= (float(i) - row["rate"]) < 1):
            df_resampled.at[index, "events"] += 1
            # row["events"] = row["events"] +1

            cnt += 1
print(f"added {cnt} times")


df_resampled.events.plot()


# df_resampled.events.rolling(10, win_type='triang').sum().plot()
df_resampled["gaussian"] = df_resampled.events.rolling(10, win_type='gaussian').sum(std=5)
df_resampled["gaussian"].plot()


df_resampled


df_resampled


dates.events.hist(bins=89)


df_resampled["gaussian"].interpolate().head(10)


from scipy.interpolate import interp1d


df_resampled["gaussian"].plot()


dt = 1
freq, spectrum = spec.amp_spec(df_resampled.events.to_numpy(), dt)
Per = np.zeros(len(freq))
Per[1:] = 1/freq[1:]


# Figure de los datos
fig, ax = plt.subplots()
ax.plot(Per, spectrum, 'r')
ax.set_xlabel('Periodo (años)')
ax.set_xlim(0,100)
ax.set_ylabel('Amplitud')


fig, ax = plt.subplots()

ax.plot(np.linspace(0,260,260),df_resampled["gaussian"], 'r')
ax.set_xlabel('Periodo (años)')
ax.set_ylabel('Amplitud')


from scipy import signal
freqs, times, spectrogram = signal.spectrogram(df_resampled["gaussian"].bfill())


freqs, psd = signal.welch(df_resampled["gaussian"].bfill())

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()


marine_extinction = df["MarineExtinction"]
nonmarine_extinction = df["NonMarineExtinction"]


data_columns_exts = [marine_extinction, nonmarine_extinction]


anoxic_event = df["AnoxicEvent"]
continental_basalt = df["ContinentalBasalt"]
sequence_boundary = df["SequenceBoundary"]
changes_spreading_rate = df["ChangesSpreadingRate"]
intraplate_volcanism = df["IntraPlateVolcanism"]

data_columns_non_exts = [anoxic_event, continental_basalt, sequence_boundary,changes_spreading_rate, intraplate_volcanism]


temp_dates = []
for i in data_columns_non_exts:
    i = cleanColumns(i)
    temp_dates.append(i)

raw_dates = []
for i in temp_dates:
    i = [x.split(",") for x in i]
    raw_dates.append(i)

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def convertFloat(lst):
    return [float(item) for item in lst]


# flatten_twice
list_dates = flatten_list(flatten_list(raw_dates))
# convert all elements to float
float_dates = convertFloat(list_dates)
# clean of Nans
cleaned_dates = [x for x in float_dates if str(x) get_ipython().getoutput("= 'nan']")

dates = pd.DataFrame(cleaned_dates)
dates.columns = ["events"]

dates_list = dates["events"].tolist()
dates_list = sorted(dates_list)


resample = np.linspace(0,260,260)
df_resampled = dates.reindex(dates.index.union(resample)).loc[resample]
df_resampled = df_resampled.reset_index()
df_resampled = df_resampled.round(1)
df_resampled.drop(columns=["events"], inplace=True)
df_resampled["events"] = 0
df_resampled.rename(columns={"index": "rate"}, inplace=True)

cnt = 0
for index, row in df_resampled.iterrows():
    
    for i in list_dates:
        
            
        if(0 <= (float(i) - row["rate"]) < 1):
            df_resampled.at[index, "events"] += 1
            # row["events"] = row["events"] +1

            cnt += 1
print(f"added {cnt} times")


df_resampled


# df_resampled.events.rolling(10, win_type='triang').sum().plot()
df_resampled["gaussian"] = df_resampled.events.rolling(10, win_type='gaussian').sum(std=5)
df_resampled["gaussian"].plot(title="Only Non-Extinction Events")



dt = 1
freq, spectrum = spec.amp_spec(df_resampled.events.to_numpy(), dt)
Per = np.zeros(len(freq))
Per[1:] = 1/freq[1:]

# Figure de los datos
fig, ax = plt.subplots()
ax.plot(Per, spectrum, 'r')
ax.set_xlabel('Periodo (años)')
ax.set_xlim(0,100)
ax.set_ylabel('Amplitud')


freqs, psd = signal.welch(df_resampled["gaussian"].bfill())

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()






