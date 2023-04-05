import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# %% Functions
def list_reader(name):
    root = "./FairTrainTest/Files/Datasets/Ignore/"
    with open(os.path.join(root, name + ".txt"), "rb") as fp:  # Unpickling
        file = pickle.load(fp)

    return file


def standardize(train_set, test_set, real_world, feat):
    rw2 = real_world.copy()
    x = train_set[feat].to_numpy().reshape(-1, 1)
    y = test_set[feat].to_numpy().reshape(-1, 1)
    rw_ = rw2[feat].to_numpy().reshape(-1, 1)
    scaler = StandardScaler().fit(x)
    train_set[feat] = scaler.transform(x)
    test_set[feat] = scaler.transform(y)
    rw2[feat] = scaler.transform(rw_)

    return train_set, test_set, rw2


# %%  Read data
train_all = pd.read_csv("./FairTrainTest/Files/Datasets/Ignore/available_data.csv")
rw = pd.read_csv("./FairTrainTest/Files/Datasets/Ignore/real_world.csv")
fair_train = list_reader("fair_train")
fair_test = list_reader("fair_test")
vsa_train = list_reader("vsa_train")
vsa_test = list_reader("vsa_test")

# %% RGB  transformer
np.array([0, 133, 13]) / 255
# %%
features = ["Porosity_Percentage", "Shc_Percentage", "Gross_Thickness_m", "TOC_weight_pct"]
dummy_feat = ["Geology 1", 'Geology 2', 'Geology 3', 'Target']
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
i = 0
for row in ax:
    for col in row:
        for train, test in zip(fair_train, fair_test):
            train, test, realw = standardize(train, test, rw, features[i])
            sns.kdeplot(data=train, x=features[i], color=(0, 0.522, 0.051, 0.05), linewidth=0.5, clip=[-6, 6],
                        ax=col)
            sns.kdeplot(data=test, x=features[i], color=(0.611, 0.611, 0.611, 0.05), linewidth=0.5, clip=[-6, 6],
                        ax=col)
            col.grid(False)
            col.xaxis.set_ticks_position('bottom')
            col.xaxis.set_tick_params(direction='in')
            col.yaxis.set_ticks_position('left')
            col.yaxis.set_tick_params(direction='in')

            col.set_xlabel(dummy_feat[i], fontdict=dict(weight='bold'))

        i += 1

i = 0
rw = pd.read_csv("./FairTrainTest/Files/Datasets/Ignore/real_world.csv")
fair_train = list_reader("fair_train")
fair_test = list_reader("fair_test")
vsa_train = list_reader("vsa_train")
vsa_test = list_reader("vsa_test")
for row in ax:
    for col in row:
        if i == 3:
            pass
        else:
            train, test, realw = standardize(fair_train[-1], fair_test[-1], rw, features[i])
            sns.kdeplot(
                data=realw, x=features[i], color=(0.612, 0.063, 0, 1), linewidth=3.0, clip=[-6, 6], ax=col,
            )
            i += 1

train, test, rw3 = standardize(fair_train[-1], fair_test[-1], rw, features[-1])
sns.kdeplot(
    data=train, x=features[-1], clip=[-6, 6], color=(0, 0.522, 0.051, 0.05), linewidth=0.5, label='Train'
)
sns.kdeplot(
    data=test, x=features[-1], clip=[-6, 6], color=(0.611, 0.611, 0.611, 0.05), linewidth=0.5, label='Test'
)
sns.kdeplot(
    data=rw3, x=features[-1], clip=[-6, 6], color=(0.612, 0.063, 0, 1), linewidth=3, label='Real-world',
)

plt.suptitle(
    'Spatial fair train-test split feature distributions',
    fontweight='bold',
    fontsize=14,
)
# handles, labels = ax[0].get_legend_handles_labels()
fig.legend(loc='lower center', ncol=3)
plt.savefig("distributions.png", dpi=500, format='png', bbox_inches='tight')
plt.show()

# %%
