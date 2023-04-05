import pickle
import numpy as np
import pandas as pd
from utils_real_set import SpatialFairSplit, plot_3_realizations, PublicationImages


# %% Function to save the lists
def save_list(name, file):
    with open(name + ".txt", "wb") as fp:  # Pickling
        pickle.dump(file, fp)


# %% Input the demonstrations
xdir = 'X_Coord_SMDA'  # the name of the column that contains the X direction
ydir = 'Y_Coord_SMDA'  # the name of the column that contains the Y direction
feature = 'TOC_weight_pct'

vario_model = {
    'nug': 999,
    'nst': 999,
    'it1': 999,
    'cc1': 999,
    'azi1': 999,
    'hmaj1': 999,
    'hmin1': 999,
    'it2': 999,
    'cc2': 999,
    'azi2': 999,
    'hmaj2': 999,
    'hmin2': 999
}

# %% read the data
training = pd.read_csv('./Files/Datasets/Ignore/available_data.csv',
                       dtype={xdir: float, ydir: float})
training.rename(columns={"UWI": "UWI1"}, inplace=True)
training.reset_index(inplace=True)  # use the well index as uwi
training = training.rename(columns={'index': 'UWI'})

real_world = pd.read_csv('./Files/Datasets/Ignore/real_world.csv',
                         dtype={xdir: float, ydir: float})
real_world.rename(columns={"UWI": "UWI1"}, inplace=True)
real_world.reset_index(inplace=True)  # use the well index as uwi
real_world = real_world.rename(columns={'index': 'UWI'})

# %% Instantiate the model
modelo = SpatialFairSplit(
    training, real_world, vario_model, xdir=xdir, ydir=ydir, number_bins=5
)
trials = 5
# %% Perform fair realizations and save results
fair_train, fair_test, fair_kvar = modelo.fair_sets_realizations(trials)
save_list("fair_train", fair_train)
save_list("fair_test", fair_test)
np.save("fair_kvar", fair_kvar)

# %% For comparison purposes, compute other sets with different cross-validation methods: the validation set approach
# (vsa) and spatial cross-validation. Moreover, get the kriging variance distribution of each. Get random and spatial
# cv sets
vsa_train, vsa_test, vsa_kvar, spatial_cv, spatial_cv_kvar = modelo.create_other_sets(realizations=trials)
save_list("vsa_train", vsa_train)
save_list("vsa_test", vsa_test)
save_list("spatial_cv", spatial_cv)
np.save("spatial_cv_kvar", spatial_cv_kvar)
np.save("vsa_kvar", vsa_kvar)

# %%  Plot the 3 methods in space
fig = plot_3_realizations(
    fair_train=fair_train,
    fair_test=fair_test,
    rand_train=vsa_train,
    rand_test=vsa_test,
    spatial_cv=spatial_cv,
    real_world_set=real_world,
    realiz=99,
    xdir=xdir,
    ydir=ydir,
    xmin=99,
    ymin=99,
    xmax=99,
    ymax=99,
)

fig.write_image("configur.png", scale=10)

# %%  Compute the divergence
fair_kvar = np.load("./Files/Datasets/Ignore/fair_kvar.npy")
vsa_kvar = np.load("./Files/Datasets/Ignore/vsa_kvar.npy")
spatial_cv_kvar = np.load("./Files/Datasets/Ignore/spatial_cv_kvar.npy")

images = PublicationImages(
    test_kvar_random=vsa_kvar,
    test_kvar_fair=fair_kvar,
    test_kvar_spatial=spatial_cv_kvar,
    rw_kvar=modelo.rw_krig_var
)
# %% Plot the kernel density estimates of the three cross-validations methods: Figures 4 and 8
figura = images.kde_plots(13.5)
figura.savefig('Real_data_kde.png', dpi=500, format='png', bbox_inches='tight')

# %% Plot the violin plots of the divergence metrics: Figures 5 and 9
plot2 = images.divergence_violins()
# plot2.savefig('real_violin.png', dpi=500, format='png')
