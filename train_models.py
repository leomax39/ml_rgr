import os
import pickle
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/moldova_cars_task_filtered.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
TARGET = "Price(euro)"
X = df.drop(columns=[TARGET])
y = df[TARGET]
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
os.makedirs("models", exist_ok=True)

results      = {}
best_params_all = {}

def objective_ridge(trial):
    alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha)),
    ])
    score = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="r2", n_jobs=-1
    ).mean()
    return score
study_ridge = optuna.create_study(direction="maximize")
study_ridge.optimize(objective_ridge, n_trials=30)
best_ridge = study_ridge.best_params
best_params_all["ML1 Ridge"] = best_ridge
model1 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=best_ridge["alpha"])),
])

model1.fit(X_train, y_train)
r2_1 = r2_score(y_test, model1.predict(X_test))
results["ML1 Ridge"] = r2_1
pickle.dump(model1, open("models/model1_ridge.pkl", "wb"))
print(f"ML1 Ridge | Лучшие параметры: {best_ridge} | R²: {r2_1:.2f}")

def objective_gb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(**params, random_state=42)),
    ])
    score = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="r2", n_jobs=-1
    ).mean()
    return score

study_gb = optuna.create_study(direction="maximize")
study_gb.optimize(objective_gb, n_trials=30)
best_gb = study_gb.best_params
best_params_all["ML2 GradientBoosting"] = best_gb
model2 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(**best_gb, random_state=42)),
])

model2.fit(X_train, y_train)
r2_2 = r2_score(y_test, model2.predict(X_test))
results["ML2 GradientBoosting"] = r2_2
pickle.dump(model2, open("models/model2_gb.pkl", "wb"))
print(f"ML2 GBM | Лучшие параметры: {best_gb} | R²: {r2_2:.2f}")

cat_features_idx = [
    list(X.columns).index(col) for col in categorical_features
]
def objective_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    model = CatBoostRegressor(
        **params,
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train, cat_features=cat_features_idx)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_catboost, n_trials=30)
best_cat = study_cat.best_params
best_params_all["ML3 CatBoost"] = best_cat
model3 = CatBoostRegressor(
    **best_cat,
    verbose=0,
    random_seed=42,
)

model3.fit(X_train, y_train, cat_features=cat_features_idx)
r2_3 = r2_score(y_test, model3.predict(X_test))
results["ML3 CatBoost"] = r2_3
model3.save_model("models/model3_catboost.cbm")
print(f"ML3 CatBoost | Лучшие параметры: {best_cat} | R²: {r2_3:.2f}")

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            **params, random_state=42, n_jobs=-1
        )),
    ])
    score = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="r2", n_jobs=-1
    ).mean()
    return score

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=30)
best_rf = study_rf.best_params
best_params_all["ML4 RandomForest"] = best_rf
model4 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        **best_rf, random_state=42, n_jobs=-1
    )),
])
model4.fit(X_train, y_train)
r2_4 = r2_score(y_test, model4.predict(X_test))
results["ML4 RandomForest"] = r2_4
pickle.dump(model4, open("models/model4_rf.pkl", "wb"))
print(f"ML4 RF | Лучшие параметры: {best_rf} | R²: {r2_4:.2f}")

def objective_stacking(trial):
    ridge_alpha = trial.suggest_float("ridge_alpha", 0.01, 50.0, log=True)
    rf_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
    rf_depth = trial.suggest_int("rf_max_depth", 3, 15)
    gb_estimators = trial.suggest_int("gb_n_estimators", 50, 300)
    gb_lr = trial.suggest_float("gb_learning_rate", 0.01, 0.2)
    gb_depth = trial.suggest_int("gb_max_depth", 2, 6)
    final_alpha = trial.suggest_float("final_alpha", 0.01, 50.0, log=True)
    estimators = [
        ("ridge", Ridge(alpha=ridge_alpha)),
        ("rf", RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            random_state=42,
            n_jobs=-1,
        )),
        ("gb", GradientBoostingRegressor(
            n_estimators=gb_estimators,
            learning_rate=gb_lr,
            max_depth=gb_depth,
            random_state=42,
        )),
    ]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=final_alpha),
        cv=3,
        n_jobs=-1,
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", stack),
    ])
    score = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="r2", n_jobs=-1
    ).mean()
    return score

study_stack = optuna.create_study(direction="maximize")
study_stack.optimize(objective_stacking, n_trials=20)
best_stack = study_stack.best_params
best_params_all["ML5 Stacking"] = best_stack
estimators_final = [
    ("ridge", Ridge(alpha=best_stack["ridge_alpha"])),
    ("rf", RandomForestRegressor(
        n_estimators=best_stack["rf_n_estimators"],
        max_depth=best_stack["rf_max_depth"],
        random_state=42,
        n_jobs=-1,
    )),
    ("gb", GradientBoostingRegressor(
        n_estimators=best_stack["gb_n_estimators"],
        learning_rate=best_stack["gb_learning_rate"],
        max_depth=best_stack["gb_max_depth"],
        random_state=42,
    )),
]
model5 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", StackingRegressor(
        estimators=estimators_final,
        final_estimator=Ridge(alpha=best_stack["final_alpha"]),
        cv=3,
        n_jobs=-1,
    )),
])

model5.fit(X_train, y_train)
r2_5 = r2_score(y_test, model5.predict(X_test))
results["ML5 Stacking"] = r2_5
pickle.dump(model5, open("models/model5_stacking.pkl", "wb"))
print(f"ML5 Stacking | Лучшие параметры: {best_stack} | R²: {r2_5:.2f}")

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
def objective_nn(trial):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    layers_list = [
        keras.layers.Input(shape=(X_train_proc.shape[1],))
    ]
    for i in range(n_layers):
        units = trial.suggest_int(f"units_{i}", 32, 256)
        layers_list.append(keras.layers.Dense(units, activation="relu"))
        layers_list.append(keras.layers.Dropout(dropout))
    layers_list.append(keras.layers.Dense(1))
    model = keras.Sequential(layers_list)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )
    model.fit(
        X_train_proc, y_train,
        epochs=40,
        batch_size=32,
        verbose=0,
    )
    preds = model.predict(X_test_proc, verbose=0).flatten()
    return r2_score(y_test, preds)

study_nn = optuna.create_study(direction="maximize")
study_nn.optimize(objective_nn, n_trials=20)
best_nn = study_nn.best_params
best_params_all["ML6 NeuralNet"] = best_nn
print(f"Лучшие параметры NN: {best_nn}")
n_layers = best_nn["n_layers"]
dropout = best_nn["dropout"]
lr = best_nn["lr"]
final_layers = [keras.layers.Input(shape=(X_train_proc.shape[1],))]
for i in range(n_layers):
    units = best_nn[f"units_{i}"]
    final_layers.append(keras.layers.Dense(units, activation="relu"))
    final_layers.append(keras.layers.Dropout(dropout))
final_layers.append(keras.layers.Dense(1))
model6 = keras.Sequential(final_layers)
model6.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss="mse",
)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)
model6.fit(
    X_train_proc, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0,
)
nn_pred = model6.predict(X_test_proc, verbose=0).flatten()
r2_6 = r2_score(y_test, nn_pred)
results["ML6 NeuralNet"] = r2_6
model6.save("models/model6_nn.keras")
pickle.dump(preprocessor, open("models/preprocessor.pkl", "wb"))
print(f"ML6 NeuralNet | R²: {r2_6:.2f}")

pickle.dump(results, open("models/results.pkl",      "wb"))
pickle.dump(best_params_all, open("models/best_params.pkl",  "wb"))
