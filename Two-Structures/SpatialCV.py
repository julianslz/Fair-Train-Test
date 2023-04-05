import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class SpatialCV:
    def __init__(self, train_set, parameters_dict, predictors, target, parameter_names, xcoor, ycoor, kfolds):
        df = train_set.copy()
        self.parameters_dict = parameters_dict
        self.predictors = predictors
        self.target = target
        self.parameter_names = parameter_names
        self.kfolds = kfolds

        scaler = StandardScaler().fit(train_set[[xcoor, ycoor]].to_numpy())
        df[['X std', 'Y std']] = scaler.transform(train_set[[xcoor, ycoor]].to_numpy())
        self.df = df

        self._kmeans()

    def _kmeans(self):
        kmeans = KMeans(n_clusters=self.kfolds, random_state=23).fit(self.df[['X std', 'Y std']])
        self.df['Labels'] = kmeans.labels_

    def _model_loss(self, parameter):
        params = {}
        for p in self.parameter_names:
            params[p] = parameter.get(p)

        rmse_per_itera = np.zeros(self.kfolds)
        for kf in range(self.kfolds):
            X = self.df.query("Labels != @kf")
            Y = self.df.query("Labels == @kf")

            X_train = X[self.predictors].to_numpy()
            y_train = X[self.target].to_numpy()
            X_test = Y[self.predictors].to_numpy()
            y_test = Y[self.target].to_numpy()

            modelo = RandomForestRegressor(random_state=42, **params)
            modelo.fit(X_train, y_train)
            predictions = modelo.predict(X_test)
            rmse_per_itera[kf] = np.sqrt(mean_squared_error(y_test, predictions))

        rmse = np.mean(rmse_per_itera)

        return rmse

    def _evaluate(self, parameters):
        return {"objective": self._model_loss(parameters)}

    def hyperparam_tuning(self, trials):
        ax_client = AxClient(verbose_logging=False)
        # create the experiment
        ax_client.create_experiment(
            parameters=self.parameters_dict,
            objective_name='objective',
            minimize=True,
        )

        with tqdm(total=trials) as pbar:
            for i in range(trials):
                parameters, trial_index = ax_client.get_next_trial()
                # Local evaluation here can be replaced with deployment to external system.
                ax_client.complete_trial(trial_index=trial_index, raw_data=self._evaluate(parameters))
                pbar.update(1)  # progress bar

        dframe = ax_client.get_trials_data_frame().sort_values('trial_index')
        dframe.sort_values('objective', ascending=True, inplace=True)
        dframe.reset_index(inplace=True, drop=True)

        return dframe


# %% variables
train = pd.read_csv("./Files/Datasets/Ignore/available_data.csv")
parameters_dictionary = [
    {
        "name": "n_estimators",
        "type": "range",
        "bounds": [18, 750],
        "value_type": "int",
    },
    {
        "name": "min_samples_leaf",
        "type": "range",
        "bounds": [7, 9],
        "value_type": "int",
    },
    {
        "name": "max_leaf_nodes",
        "type": "range",
        "bounds": [4, 22],
        "value_type": "int",
    },
    {
        "name": "max_depth",
        "type": "range",
        "bounds": [10, 60],
        "value_type": "int",
    },
]
features = ['Porosity_Percentage', 'Shc_Percentage', 'Gross_Thickness_m']
target_ = 'TOC_weight_pct'
all_params = ["n_estimators", "min_samples_leaf", "max_leaf_nodes", "max_depth"]
# %%
scv = SpatialCV(
    train, parameters_dictionary, features, target_, all_params, 'X_Coord_SMDA', 'Y_Coord_SMDA', 4
)

# %%
result = scv.hyperparam_tuning(20)

# %%
rw = pd.read_csv("./Files/Datasets/Ignore/real_world.csv")
X_train_ = train[features].to_numpy()
y_train_ = train[target_].to_numpy()
X_test_ = rw[features].to_numpy()
y_test_ = rw[target_].to_numpy()

best_params = {
    "n_estimators": 271,
    "min_samples_leaf": 7,
    "max_leaf_nodes": 4,
    "max_depth": 31,
    "random_state": 42
}

model = RandomForestRegressor(**best_params)
model.fit(X_train_, y_train_)
predict = model.predict(X_test_)
rmse_scv = np.sqrt(mean_squared_error(y_test_, predict))
