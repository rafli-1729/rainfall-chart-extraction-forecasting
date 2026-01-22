from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    clone
)

from xgboost import (
    XGBRegressor,
    XGBClassifier
)

import pandas as pd
import numpy as np

from src.config import config
from src.features import (
    TimeFeatures,
    TemperatureFeatures,
    WindRainFeatures,
    LagFeatures,
    RollingStatsFeatures,
    CyclicalInteractionFeatures,
    DropFeatures,
    DebugTransformer,
    StructuralWeatherImputer,
    AddExternalFeatures,
)

from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np

class TwoStageRainfallModel(BaseEstimator, RegressorMixin):
    def __init__(self, classifier, regressor, rain_threshold=0.0):
        self.classifier = classifier
        self.regressor = regressor
        self.rain_threshold = rain_threshold

    def fit(self, X, y):
        self.clf_ = clone(self.classifier)
        self.reg_ = clone(self.regressor)

        rain_flag = (y > self.rain_threshold).astype(int)

        self.clf_.fit(X, rain_flag)

        mask = (rain_flag == 1).to_numpy()

        if mask.sum() == 0:
            self.has_regressor_ = False
            return self

        self.has_regressor_ = True
        self.reg_.fit(X[mask], y.to_numpy()[mask])

        return self


    def predict(self, X):
        p_rain = self.clf_.predict_proba(X)[:, 1]

        if not self.has_regressor_:
            return np.zeros(len(X))

        rain_pred = self.reg_.predict(X)
        return p_rain * rain_pred


def build_feature_pipeline(external_df: pd.DataFrame):
    return Pipeline(steps=[
        ("structural_imputer", StructuralWeatherImputer()),
        ("external_features", AddExternalFeatures(external_df)),
        ("time", TimeFeatures()),
        ("temp", TemperatureFeatures()),
        ("wind_rain", WindRainFeatures()),
        ("lag", LagFeatures(lag_days=1)),
        ("rolling", RollingStatsFeatures()),
        ("cyclical", CyclicalInteractionFeatures()),
        ("drop", DropFeatures(['date']))
    ])


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ['location']),
            ("num", "passthrough",
            config.features.rain_extreme_columns +
            config.features.meteorogical_columns),
        ],
        remainder="drop"
    )


def build_classifier(xgb_params=None):
    default_params = dict(
        objective="binary:logistic",
        scale_pos_weight=1.25,
        max_depth=3,
        min_child_weight=5,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    if xgb_params:
        default_params.update(xgb_params)

    return XGBClassifier(**default_params)


def build_regressor(xgb_params=None, transform_target=False):
    default_params = dict(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    if xgb_params:
        default_params.update(xgb_params)

    model = XGBRegressor(**default_params)

    if transform_target:
        model = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        )

    return model

def build_model(
    model_type: str = "regressor",
    xgb_params=None,
    transform_target=False,
    classifier_params=None,
    rain_threshold=0.0
):
    if model_type == "regressor":
        return build_regressor(
            xgb_params=xgb_params,
            transform_target=transform_target
        )

    if model_type == "classifier":
        return build_classifier(xgb_params=classifier_params)

    if model_type == "two_stage":
        clf = build_classifier(xgb_params=classifier_params)
        reg = build_regressor(
            xgb_params=xgb_params,
            transform_target=transform_target
        )

        return TwoStageRainfallModel(
            classifier=clf,
            regressor=reg,
            rain_threshold=rain_threshold
        )

    raise ValueError(f"Unknown model_type: {model_type}")

def build_pipeline(
    external_df: pd.DataFrame,
    model_type: str ="two_stage",
    transform_target: bool = False
):
    return Pipeline(steps=[
        ("features", build_feature_pipeline(external_df)),
        ("preprocess", build_preprocessor()),
        ("model", build_model(
            model_type=model_type,
            transform_target=transform_target,
            rain_threshold=0.1
        ))
    ])


def inference_data(
    model,
    data: pd.DataFrame,
    clip_negative: bool = True,
    return_dataframe: bool = True
):
    preds = model.predict(data)

    if clip_negative:
        preds = np.clip(preds, 0, None)

    if return_dataframe:
        result = data.copy()
        result["predicted_daily_rainfall_mm"] = preds
        return result

    return preds