"""Registry of every estimator the workbench can train.

Each entry declares its scikit-learn class, which tasks it can serve, and the
hyperparameters worth exposing — with types, defaults, sensible ranges and a
small default search grid.  The GUI builds its parameter form purely from these
declarations, so adding a model here makes it appear in the interface with no
GUI changes at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.base import BaseEstimator, ClassifierMixin, clone


def _has(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # find_spec raises rather than returning None for a module that is in
        # sys.modules but half-initialised; treat that as "not usable".
        return False


HAVE_XGBOOST = _has("xgboost")
HAVE_LIGHTGBM = _has("lightgbm")
HAVE_TORCH = _has("torch")


@dataclass
class Param:
    name: str
    label: str
    kind: str = "float"          # int | float | bool | choice | str | none
    default: object = None
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list = field(default_factory=list)
    grid: list = field(default_factory=list)
    help: str = ""
    advanced: bool = False

    def coerce(self, value):
        """Turn a GUI string into the Python type scikit-learn expects."""
        if value is None or value == "":
            return None
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y", "on")
            return bool(value)
        if self.kind == "int":
            return int(float(value))
        if self.kind == "float":
            return float(value)
        if self.kind == "choice":
            for c in self.choices:
                if str(c).lower() == str(value).lower():
                    return c
            return value
        if self.kind == "none":
            return None
        return value


def P(name, label, kind="float", default=None, low=None, high=None, step=None, log=False,
      choices=None, grid=None, help="", advanced=False) -> Param:
    return Param(name, label, kind, default, low, high, step, log, choices or [], grid or [], help, advanced)


_RS = P("random_state", "random state", "int", 42, advanced=True, help="fixed for reproducibility")
_MAXFEAT = P("max_features", "features per split", "choice", None, choices=[None, "sqrt", "log2", 0.5, 1.0])


# --------------------------------------------------------------------------
# deep-learning parameters
#
# These are shared by every neural network in the registry, and the split between
# the two helpers is deliberate: only ``hidden_layers`` and ``learning_rate`` carry
# a ``grid``, because those are the two knobs actually worth searching.  Everything
# else is left out of the default sweep on purpose — a grid over the optimiser,
# the activation and the batch size as well would be hundreds of network fits, and
# this machine has four cores.
# --------------------------------------------------------------------------


def _dl_architecture_params() -> list[Param]:
    return [
        P("hidden_layers", "hidden layers", "str", "64, 32",
          grid=["32", "64, 32", "128, 64"],
          help="Comma-separated layer sizes, e.g. 128, 64, 32.  Wider and deeper is more "
               "expressive and slower — on a 4-core CPU start small and grow only if the "
               "model clearly underfits."),
        P("activation", "activation", "choice", "relu",
          choices=["relu", "gelu", "tanh", "leaky_relu", "silu"]),
        P("dropout", "dropout", "float", 0.1, 0.0, 0.9,
          help="Randomly zero this fraction of hidden units while training.  This is the main "
               "defence against overfitting on a small dataset; 0.0 disables it."),
        P("batch_norm", "batch normalisation", "bool", False),
        P("use_embeddings", "embeddings for categorical columns", "bool", True,
          help="Replace each categorical one-hot block with a learned embedding.  Cheaper and "
               "usually better on columns with many categories; turn it off to get the plain "
               "one-hot network instead."),
        P("embedding_dim", "embedding width (0 = auto)", "int", 0, 0, 64, advanced=True),
    ]


def _dl_training_params(sweep_learning_rate: bool = True) -> list[Param]:
    return [
        P("learning_rate", "learning rate", "float", 1e-3, 1e-6, 1.0, log=True,
          grid=[1e-3, 3e-3] if sweep_learning_rate else [],
          help="Step size for the optimiser.  The single most influential knob, and the first "
               "thing to lower if the loss curve jumps around."),
        P("optimizer", "optimiser", "choice", "adam",
          choices=["adam", "adamw", "sgd", "rmsprop"]),
        P("weight_decay", "weight decay", "float", 1e-4, 0.0, 0.1, log=True, advanced=True),
        P("epochs", "max epochs", "int", 120, 1, 5000,
          help="Upper bound on passes over the data.  Early stopping usually ends training long "
               "before this."),
        P("batch_size", "batch size", "int", 64, 4, 4096,
          help="Rows per gradient step.  Smaller batches update more often but are noisier and "
               "slower per epoch."),
        P("patience", "early-stopping patience", "int", 12, 0, 200,
          help="Stop after this many epochs with no improvement on the validation split.  "
               "0 disables early stopping and runs every epoch."),
        P("val_fraction", "validation fraction", "float", 0.15, 0.0, 0.5, advanced=True,
          help="Slice of the training data held out to decide when to stop.  0 disables it, "
               "and then the training loss is used instead."),
        P("threads", "CPU threads", "int", 4, 1, 64,
          help="How many CPU threads PyTorch may use.  Networks this small are often *faster* "
               "with a few threads than with every logical core."),
        P("max_seconds", "time budget per fit (s)", "int", 120, 0, 100000,
          help="Hard wall-clock limit for one fit.  A fit that reaches it keeps the best weights "
               "so far instead of hanging; 0 means no limit."),
        _RS,
    ]



@dataclass
class ModelSpec:
    key: str
    label: str
    tasks: tuple[str, ...]          # regression | binary | multiclass | unsupervised
    family: str
    params: list[Param] = field(default_factory=list)
    notes: str = ""
    optional: str = ""              # package this model depends on, when not sklearn
    needs_numeric_labels: bool = False   # 0..n-1 only: wrap in NumericLabelClassifier

    def default_params(self) -> dict:
        return {p.name: p.default for p in self.params if p.default is not None}

    def suggested_grid(self) -> dict:
        out = {}
        for p in self.params:
            if p.grid and len(p.grid) > 1:
                out[p.name] = list(p.grid)
        return out


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

MODELS: list[ModelSpec] = [
    # ---------------- linear regression ----------------
    ModelSpec("linear", "Linear regression", ("regression",), "linear", [
        P("fit_intercept", "fit intercept", "bool", True, advanced=True),
    ], "Baseline. Assumes a straight-line relationship between each feature and the target."),
    ModelSpec("ridge", "Ridge regression", ("regression",), "linear", [
        P("alpha", "regularisation strength (alpha)", "float", 1.0, 1e-6, 1e4, log=True,
          grid=[0.01, 0.1, 1.0, 10.0, 100.0]),
        P("fit_intercept", "fit intercept", "bool", True, advanced=True),
    ], "Linear model with L2 penalty — tames unstable coefficients when features correlate."),
    ModelSpec("lasso", "Lasso regression", ("regression",), "linear", [
        P("alpha", "regularisation strength (alpha)", "float", 0.1, 1e-6, 1e4, log=True, grid=[0.001, 0.01, 0.1, 1.0]),
        P("max_iter", "max iterations", "int", 5000, advanced=True),
    ], "L1 penalty: drives useless coefficients to exactly zero, so it doubles as feature selection."),
    ModelSpec("elasticnet", "Elastic net", ("regression",), "linear", [
        P("alpha", "strength (alpha)", "float", 0.1, 1e-6, 1e3, log=True, grid=[0.001, 0.01, 0.1, 1.0]),
        P("l1_ratio", "L1 ratio", "float", 0.5, 0, 1, grid=[0.1, 0.5, 0.9]),
        P("max_iter", "max iterations", "int", 5000, advanced=True),
    ], "Mix of Lasso and Ridge; useful when features are correlated *and* numerous."),
    ModelSpec("bayesian_ridge", "Bayesian ridge", ("regression",), "linear", [
        P("alpha_1", "alpha prior shape", "float", 1e-6, advanced=True),
        P("lambda_1", "lambda prior shape", "float", 1e-6, advanced=True),
    ], "Ridge where the penalty is learned from the data instead of supplied."),
    ModelSpec("huber", "Huber regressor", ("regression",), "linear", [
        P("epsilon", "outlier threshold (epsilon)", "float", 1.35, 1.0, 3.0, grid=[1.1, 1.35, 2.0]),
        P("alpha", "penalty", "float", 1e-4, 1e-8, 1.0, log=True, advanced=True),
    ], "Least-squares that ignores extreme outliers — good when a few lab tests failed."),
    ModelSpec("pls", "Partial least squares", ("regression",), "linear", [
        P("n_components", "components", "int", 2, 1, 50, grid=[2, 5, 10]),
    ], "Projects correlated features onto a few latent axes before regressing."),

    # ---------------- neighbours & kernels ----------------
    ModelSpec("knn_reg", "K-nearest neighbours", ("regression",), "instance", [
        P("n_neighbors", "neighbours (k)", "int", 5, 1, 100, grid=[3, 5, 10, 20]),
        P("weights", "weighting", "choice", "uniform", choices=["uniform", "distance"]),
        P("p", "distance power (p)", "choice", 2, choices=[1, 2]),
    ], "Averages the k most similar historical records. Needs scaling (it is on by default)."),
    ModelSpec("svr", "Support vector regression", ("regression",), "kernel", [
        P("C", "C (error tolerance)", "float", 1.0, 1e-3, 1e4, log=True, grid=[0.1, 1.0, 10.0, 100.0]),
        P("epsilon", "epsilon tube", "float", 0.1, 0, 10, grid=[0.01, 0.1, 0.5]),
        P("kernel", "kernel", "choice", "rbf", choices=["rbf", "linear", "poly", "sigmoid"]),
        P("gamma", "gamma", "choice", "scale", choices=["scale", "auto"]),
    ], "Strong on smooth non-linear responses; slow on large datasets."),

    # ---------------- trees ----------------
    ModelSpec("decision_tree_reg", "Decision tree", ("regression",), "tree", [
        P("criterion", "split quality", "choice", "squared_error", choices=["squared_error", "friedman_mse", "absolute_error", "poisson"]),
        P("max_depth", "max depth", "choice", None, choices=[None, 3, 5, 8, 12, 20], grid=[3, 5, 8, None]),
        P("min_samples_split", "min samples to split", "int", 2, 2, 50, grid=[2, 5, 10]),
        P("min_samples_leaf", "min samples per leaf", "int", 1, 1, 50, grid=[1, 2, 5]),
        _RS,
    ], "Easy to read but overfits on its own — a stepping stone to the ensembles below."),
    ModelSpec("random_forest_reg", "Random forest", ("regression",), "ensemble", [
        P("n_estimators", "trees", "int", 300, 10, 3000, grid=[100, 300, 600]),
        P("max_depth", "max depth", "choice", None, choices=[None, 5, 10, 20, 30], grid=[None, 10, 20]),
        P("min_samples_leaf", "min samples per leaf", "int", 1, 1, 30, grid=[1, 2, 4]),
        _MAXFEAT,
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "The safest strong default. Averages many decorrelated trees, so it is hard to overfit badly."),
    ModelSpec("extra_trees_reg", "Extra trees", ("regression",), "ensemble", [
        P("n_estimators", "trees", "int", 300, 10, 3000, grid=[100, 300, 600]),
        P("max_depth", "max depth", "choice", None, choices=[None, 5, 10, 20, 30]),
        P("min_samples_leaf", "min samples per leaf", "int", 1, 1, 30),
        _MAXFEAT,
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "Like a random forest but splits are randomised further — faster, slightly more variance."),
    ModelSpec("gradient_boosting_reg", "Gradient boosting (sklearn)", ("regression",), "boosting", [
        P("n_estimators", "boosting rounds", "int", 200, 10, 3000, grid=[100, 300, 600]),
        P("learning_rate", "learning rate", "float", 0.1, 0.001, 1.0, log=True, grid=[0.01, 0.05, 0.1, 0.2]),
        P("max_depth", "max depth", "int", 3, 1, 15, grid=[2, 3, 5]),
        P("subsample", "row subsample", "float", 1.0, 0.1, 1.0, grid=[0.8, 1.0]),
        _RS,
    ], "Sequential boosting — often the most accurate of the sklearn ensembles, but slower."),
    ModelSpec("hist_gb_reg", "Histogram gradient boosting", ("regression",), "boosting", [
        P("max_iter", "boosting rounds", "int", 300, 10, 5000, grid=[200, 500]),
        P("learning_rate", "learning rate", "float", 0.1, 0.001, 1.0, log=True, grid=[0.05, 0.1, 0.2]),
        P("max_leaf_nodes", "max leaves", "int", 31, 2, 255, grid=[15, 31, 63]),
        P("min_samples_leaf", "min samples per leaf", "int", 20, 1, 200, grid=[10, 20, 50]),
        P("l2_regularization", "L2 penalty", "float", 0.0, 0, 10, grid=[0.0, 1.0]),
        _RS,
    ], "sklearn's fast boosted-tree implementation. Handles large data well and needs little tuning."),
    ModelSpec("adaboost_reg", "AdaBoost", ("regression",), "boosting", [
        P("n_estimators", "estimators", "int", 100, 10, 2000, grid=[50, 100, 300]),
        P("learning_rate", "learning rate", "float", 1.0, 0.001, 5.0, log=True, grid=[0.1, 0.5, 1.0]),
        _RS,
    ], "Reweights mispredicted rows each round — sensitive to outliers."),

    # ---------------- optional boosters ----------------
    ModelSpec("xgboost_reg", "XGBoost", ("regression",), "boosting", [
        P("n_estimators", "boosting rounds", "int", 400, 10, 5000, grid=[200, 400, 800]),
        P("learning_rate", "learning rate", "float", 0.05, 0.001, 1.0, log=True, grid=[0.02, 0.05, 0.1]),
        P("max_depth", "max depth", "int", 6, 1, 15, grid=[3, 6, 9]),
        P("subsample", "row subsample", "float", 0.9, 0.1, 1.0, grid=[0.7, 0.9]),
        P("colsample_bytree", "column subsample", "float", 0.9, 0.1, 1.0, grid=[0.7, 0.9]),
        P("reg_lambda", "L2 penalty", "float", 1.0, 0, 20, grid=[0.5, 1.0, 5.0]),
        P("min_child_weight", "min child weight", "float", 1.0, 0, 20),
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "Gradient boosting with regularisation and column subsampling.", optional="xgboost"),
    ModelSpec("lightgbm_reg", "LightGBM", ("regression",), "boosting", [
        P("n_estimators", "boosting rounds", "int", 400, 10, 5000, grid=[200, 400, 800]),
        P("learning_rate", "learning rate", "float", 0.05, 0.001, 1.0, log=True, grid=[0.02, 0.05, 0.1]),
        P("num_leaves", "leaves", "int", 31, 2, 500, grid=[15, 31, 63]),
        P("max_depth", "max depth", "choice", -1, choices=[-1, 3, 5, 8, 12]),
        P("min_child_samples", "min samples per leaf", "int", 20, 1, 200, grid=[10, 20, 40]),
        P("subsample", "bagging fraction", "float", 0.9, 0.1, 1.0),
        P("colsample_bytree", "feature fraction", "float", 0.9, 0.1, 1.0),
        P("reg_lambda", "L2 penalty", "float", 0.0, 0, 20, grid=[0.0, 1.0]),
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "Leaf-wise boosting; very fast on wide data. Small datasets need small num_leaves.",
       optional="lightgbm"),

    # ---------------- classification ----------------
    ModelSpec("logistic", "Logistic regression", ("binary", "multiclass"), "linear", [
        P("C", "inverse regularisation (C)", "float", 1.0, 1e-4, 1e4, log=True, grid=[0.01, 0.1, 1.0, 10.0]),
        P("solver", "solver", "choice", "lbfgs", choices=["lbfgs", "liblinear", "saga", "newton-cg"]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        P("max_iter", "max iterations", "int", 1000, advanced=True),
    ], "The interpretable workhorse: coefficients read as log-odds per unit of each feature. "
       "L2 penalty, which is what you want when the features correlate."),
    ModelSpec("logistic_l1", "Logistic regression (L1)", ("binary", "multiclass"), "linear", [
        P("C", "inverse regularisation (C)", "float", 1.0, 1e-4, 1e4, log=True, grid=[0.01, 0.1, 1.0, 10.0]),
        P("l1_ratio", "L1 / L2 mix", "float", 1.0, 0.0, 1.0, grid=[0.0, 0.5, 1.0],
          help="1.0 = pure L1, 0.0 = pure L2, in between = elastic net.  scikit-learn "
               "expresses the penalty this way since 'penalty' was deprecated in 1.8"),
        P("solver", "solver", "choice", "saga", choices=["saga", "liblinear"]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        P("max_iter", "max iterations", "int", 2000, advanced=True),
    ], "L1 penalty: drives useless coefficients to exactly zero, so it doubles as feature selection."),
    ModelSpec("ridge_clf", "Ridge classifier", ("binary", "multiclass"), "linear", [
        P("alpha", "regularisation", "float", 1.0, 1e-6, 1e3, log=True, grid=[0.1, 1.0, 10.0]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
    ], "Fast linear classifier; no probabilities, only decisions."),
    ModelSpec("linear_svc", "Linear SVM", ("binary", "multiclass"), "kernel", [
        P("C", "C", "float", 1.0, 1e-3, 1e4, log=True, grid=[0.01, 0.1, 1.0, 10.0]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        P("max_iter", "max iterations", "int", 5000, advanced=True),
    ], "Wide-margin linear boundary; robust when classes are separable."),
    ModelSpec("knn_clf", "K-nearest neighbours", ("binary", "multiclass"), "instance", [
        P("n_neighbors", "neighbours (k)", "int", 5, 1, 100, grid=[3, 5, 10, 20]),
        P("weights", "weighting", "choice", "uniform", choices=["uniform", "distance"]),
        P("p", "distance power (p)", "choice", 2, choices=[1, 2]),
    ], "Votes among the closest training rows. Very sensitive to feature scaling."),
    ModelSpec("svc", "Support vector classifier", ("binary", "multiclass"), "kernel", [
        P("C", "C", "float", 1.0, 1e-3, 1e4, log=True, grid=[0.1, 1.0, 10.0, 100.0]),
        P("kernel", "kernel", "choice", "rbf", choices=["rbf", "linear", "poly", "sigmoid"]),
        P("gamma", "gamma", "choice", "scale", choices=["scale", "auto"]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        P("probability", "estimate probabilities", "bool", True, advanced=True),
    ], "Non-linear boundaries via kernels. Probabilities are cross-validated, so fits are slower."),
    ModelSpec("decision_tree_clf", "Decision tree", ("binary", "multiclass"), "tree", [
        P("criterion", "split quality", "choice", "gini", choices=["gini", "entropy", "log_loss"]),
        P("max_depth", "max depth", "choice", None, choices=[None, 3, 5, 8, 12, 20], grid=[3, 5, 8, None]),
        P("min_samples_leaf", "min samples per leaf", "int", 1, 1, 50, grid=[1, 2, 5]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        _RS,
    ], "Readable rules you can show a reviewer — at the cost of accuracy."),
    ModelSpec("random_forest_clf", "Random forest", ("binary", "multiclass"), "ensemble", [
        P("n_estimators", "trees", "int", 300, 10, 3000, grid=[100, 300, 600]),
        P("max_depth", "max depth", "choice", None, choices=[None, 5, 10, 20, 30], grid=[None, 10, 20]),
        P("min_samples_leaf", "min samples per leaf", "int", 1, 1, 30, grid=[1, 2, 4]),
        _MAXFEAT,
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced", "balanced_subsample"]),
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "Strong out-of-the-box classifier and gives reliable feature importances."),
    ModelSpec("extra_trees_clf", "Extra trees", ("binary", "multiclass"), "ensemble", [
        P("n_estimators", "trees", "int", 300, 10, 3000, grid=[100, 300, 600]),
        P("max_depth", "max depth", "choice", None, choices=[None, 5, 10, 20, 30]),
        P("min_samples_leaf", "min samples per leaf", "int", 1, 1, 30),
        _MAXFEAT,
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "Faster than a forest, marginally noisier."),
    ModelSpec("gradient_boosting_clf", "Gradient boosting (sklearn)", ("binary", "multiclass"), "boosting", [
        P("n_estimators", "boosting rounds", "int", 200, 10, 2000, grid=[100, 300]),
        P("learning_rate", "learning rate", "float", 0.1, 0.001, 1.0, log=True, grid=[0.05, 0.1, 0.2]),
        P("max_depth", "max depth", "int", 3, 1, 15, grid=[2, 3, 5]),
        P("subsample", "row subsample", "float", 1.0, 0.1, 1.0),
        _RS,
    ], "Boosting for classification; often edges out the forest on tabular data."),
    ModelSpec("hist_gb_clf", "Histogram gradient boosting", ("binary", "multiclass"), "boosting", [
        P("max_iter", "boosting rounds", "int", 300, 10, 3000, grid=[200, 500]),
        P("learning_rate", "learning rate", "float", 0.1, 0.001, 1.0, log=True, grid=[0.05, 0.1, 0.2]),
        P("max_leaf_nodes", "max leaves", "int", 31, 2, 255, grid=[15, 31, 63]),
        P("l2_regularization", "L2 penalty", "float", 0.0, 0, 10, grid=[0.0, 1.0]),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"]),
        _RS,
    ], "Fast, handles missing values natively, very few knobs to get wrong."),
    ModelSpec("adaboost_clf", "AdaBoost", ("binary", "multiclass"), "boosting", [
        P("n_estimators", "estimators", "int", 100, 10, 2000, grid=[50, 100, 300]),
        P("learning_rate", "learning rate", "float", 1.0, 0.001, 5.0, log=True, grid=[0.1, 0.5, 1.0]),
        _RS,
    ], "Boosts shallow trees by reweighting hard rows."),
    ModelSpec("gaussian_nb", "Gaussian naive Bayes", ("binary", "multiclass"), "probabilistic", [
        P("var_smoothing", "variance smoothing", "float", 1e-9, 1e-12, 1e-3, log=True),
    ], "Instant baseline; assumes features are independent and roughly normal."),
    ModelSpec("xgboost_clf", "XGBoost", ("binary", "multiclass"), "boosting", [
        P("n_estimators", "boosting rounds", "int", 400, 10, 5000, grid=[200, 400, 800]),
        P("learning_rate", "learning rate", "float", 0.05, 0.001, 1.0, log=True, grid=[0.02, 0.05, 0.1]),
        P("max_depth", "max depth", "int", 6, 1, 15, grid=[3, 6, 9]),
        P("subsample", "row subsample", "float", 0.9, 0.1, 1.0, grid=[0.7, 0.9]),
        P("colsample_bytree", "column subsample", "float", 0.9, 0.1, 1.0),
        P("reg_lambda", "L2 penalty", "float", 1.0, 0, 20, grid=[0.5, 1.0, 5.0]),
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "State-of-the-art on many tabular benchmarks.", optional="xgboost",
       needs_numeric_labels=True),
    ModelSpec("lightgbm_clf", "LightGBM", ("binary", "multiclass"), "boosting", [
        P("n_estimators", "boosting rounds", "int", 400, 10, 5000, grid=[200, 400, 800]),
        P("learning_rate", "learning rate", "float", 0.05, 0.001, 1.0, log=True, grid=[0.02, 0.05, 0.1]),
        P("num_leaves", "leaves", "int", 31, 2, 500, grid=[15, 31, 63]),
        P("min_child_samples", "min samples per leaf", "int", 20, 1, 200, grid=[10, 20, 40]),
        P("subsample", "bagging fraction", "float", 0.9, 0.1, 1.0),
        P("colsample_bytree", "feature fraction", "float", 0.9, 0.1, 1.0),
        P("reg_lambda", "L2 penalty", "float", 0.0, 0, 20),
        P("n_jobs", "parallel jobs", "int", -1, advanced=True),
        _RS,
    ], "Fast boosted trees, strong on wide or noisy clinical/sensor data.", optional="lightgbm"),

    # ---------------- unsupervised ----------------
    ModelSpec("kmeans", "K-means", ("unsupervised",), "cluster", [
        P("n_clusters", "clusters (k)", "int", 3, 2, 50, grid=[2, 3, 4, 5, 8]),
        P("init", "initialisation", "choice", "k-means++", choices=["k-means++", "random"]),
        P("n_init", "restarts", "int", 10, 1, 50),
        P("max_iter", "max iterations", "int", 300, advanced=True),
        _RS,
    ], "Groups rows into k clusters of similar behaviour. Check the elbow and silhouette plots."),
    ModelSpec("dbscan", "DBSCAN", ("unsupervised",), "cluster", [
        P("eps", "neighbourhood radius (eps)", "float", 0.5, 0.01, 50, log=True, grid=[0.1, 0.3, 0.5, 1.0]),
        P("min_samples", "min samples per core point", "int", 5, 2, 100, grid=[3, 5, 10]),
    ], "Finds arbitrarily shaped clusters and labels stragglers as noise. Needs scaling."),
    ModelSpec("agglomerative", "Agglomerative (hierarchical)", ("unsupervised",), "cluster", [
        P("n_clusters", "clusters", "int", 3, 2, 50, grid=[2, 3, 4, 5, 8]),
        P("linkage", "linkage", "choice", "ward", choices=["ward", "complete", "average", "single"]),
        P("metric", "distance metric", "choice", "euclidean", choices=["euclidean", "manhattan", "cosine", "cityblock"]),
    ], "Builds a hierarchy you can read off a dendrogram — good for small datasets."),
    ModelSpec("pca", "Principal component analysis", ("unsupervised", "transform"), "decomposition", [
        P("n_components", "components", "int", 2, 1, 100, grid=[2, 3, 5, 10]),
        P("whiten", "whiten", "bool", False),
        _RS,
    ], "Compresses correlated features into a few uncorrelated axes; useful for plotting."),
    ModelSpec("tsne", "t-SNE (visualisation)", ("unsupervised", "transform"), "manifold", [
        P("n_components", "components", "choice", 2, choices=[2, 3]),
        P("perplexity", "perplexity", "float", 30.0, 5, 100, grid=[10, 30, 50]),
        P("learning_rate", "learning rate", "choice", "auto", choices=["auto", 200.0, 500.0]),
        P("max_iter", "iterations", "int", 1000, 250, 5000),
        _RS,
    ], "Non-linear embedding for 2-D pictures only. Slow above a few thousand rows — it samples if asked."),

    # ---------------- deep learning (CPU-only PyTorch) ----------------
    ModelSpec("torch_mlp_reg", "Neural network (MLP)", ("regression",), "deep", [
        *_dl_architecture_params(),
        P("loss", "loss", "choice", "mse", choices=["mse", "huber"],
          help="'huber' behaves like squared error for most rows but stops a handful of extreme "
               "ones from dominating the fit."),
        P("standardize_target", "standardise the target", "bool", True,
          help="Train against a zero-mean unit-variance target and undo it on prediction.  Almost "
               "always makes training more stable."),
        *_dl_training_params(),
    ], "A feed-forward network with a learned embedding per categorical column. It can pick up "
       "interactions and curved relationships that no linear model can express, without being "
       "told their shape. Trained on CPU with early stopping, a time budget and a pinned thread "
       "count — start small and grow only if it underfits.", optional="torch"),
    ModelSpec("torch_mlp_clf", "Neural network (MLP)", ("binary", "multiclass"), "deep", [
        *_dl_architecture_params(),
        P("class_weight", "class weight", "choice", None, choices=[None, "balanced"],
          help="'balanced' weights the loss by inverse class frequency — the usual fix when one "
               "class is rare."),
        P("label_smoothing", "label smoothing", "float", 0.0, 0.0, 0.3, advanced=True,
          help="Softens the training targets slightly; a mild regulariser."),
        *_dl_training_params(),
    ], "A feed-forward classifier with embeddings for categorical columns, class weighting for "
       "imbalanced targets and early stopping.  Probabilities come from a softmax over the "
       "output layer.", optional="torch"),
    ModelSpec("torch_autoencoder", "Autoencoder (deep clustering)", ("unsupervised", "transform"),
              "deep", [
        P("hidden_layers", "encoder layers", "str", "64, 32", grid=["32", "64, 32"],
          help="Encoder layer sizes; the decoder mirrors them."),
        P("latent_dim", "latent dimensions", "int", 8, 1, 256, grid=[2, 4, 8],
          help="Size of the learned compression — these are the axes the cluster and projection "
               "plots use."),
        P("n_clusters", "clusters", "int", 3, 0, 50,
          help="K-means is run on the latent code and its labels feed the cluster plots.  "
               "0 disables clustering and leaves only the embedding."),
        P("activation", "activation", "choice", "relu",
          choices=["relu", "gelu", "tanh", "leaky_relu", "silu"]),
        P("dropout", "dropout", "float", 0.0, 0.0, 0.9),
        P("batch_norm", "batch normalisation", "bool", False),
        *_dl_training_params(sweep_learning_rate=False),
    ], "Compresses the prepared matrix into a small latent code and reconstructs it. A learned, "
       "non-linear alternative to PCA: cluster the code, plot it, or read the reconstruction "
       "error as an anomaly score.", optional="torch"),
]

MODEL_INDEX = {m.key: m for m in MODELS}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_INDEX:
        raise KeyError(f"unknown model {key!r}")
    return MODEL_INDEX[key]


def models_for_task(task: str) -> list[ModelSpec]:
    if task is None:
        return list(MODELS)
    want = {"binary": "binary", "multiclass": "multiclass", "regression": "regression", "unsupervised": "unsupervised"}.get(task, task)
    return [m for m in MODELS if want in m.tasks]


def is_available(spec: ModelSpec) -> bool:
    """Whether the optional library a model needs can actually be used.

    ``find_spec`` alone is not enough for torch: a torch that is *installed but
    broken* (a missing shared library, a wheel for the wrong Python) passes
    ``find_spec`` and then explodes inside ``fit``.  Importing it here greys the
    network out instead, and ``runtime.torch_problem()`` can say why.
    """
    if not spec.optional:
        return True
    if spec.optional == "torch":
        from .dl import runtime

        return runtime.torch_available()
    return _has(spec.optional)


def make_model(key: str, params: dict | None = None):
    """Instantiate the estimator for ``key``, keeping only parameters it accepts.

    Parameters the installed library version does not recognise are dropped and
    recorded in :data:`DROPPED_PARAMS` rather than raising — library keyword sets
    drift between releases, and a silently-dropped knob is far better than a
    crash in the middle of a sweep.
    """
    spec = get_model_spec(key)
    params = dict(params or {})
    allowed = {p.name: p for p in spec.params}
    clean = {}
    for name, value in params.items():
        if name not in allowed:
            continue
        coerced = allowed[name].coerce(value)
        if coerced is not None:
            clean[name] = coerced
    builder = _CONSTRUCTORS[key]
    estimator = builder(clean)
    if spec.needs_numeric_labels:
        estimator = NumericLabelClassifier(estimator)
    return estimator


class NumericLabelClassifier(BaseEstimator, ClassifierMixin):
    """Give an estimator that insists on 0..n-1 integer labels (XGBoost) what it wants.

    XGBoost refuses a multiclass target whose values are text ("low"/"mid"/"high" —
    the common case in this workbench).  This wrapper encodes the target on the way
    in and restores the original vocabulary on the way out, so ``predict``,
    ``predict_proba`` and every metric still speak the user's labels.  The fitted
    inner estimator stays reachable, so ``feature_importances_`` and ``coef_`` work
    as if it were not wrapped at all.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.encoder_.transform(y), **fit_params)
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def __getattr__(self, name):
        # anything the inner model learned (feature_importances_, coef_, n_features_in_)
        if name.startswith("__"):
            raise AttributeError(name)
        for attribute in ("estimator_", "estimator"):
            inner = self.__dict__.get(attribute)
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
        raise AttributeError(name)


DROPPED_PARAMS: list[str] = []


def _filter_signature(key: str, cls, params: dict) -> dict:
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return params
    params_iter = list(sig.parameters.values())
    if any(p.kind == p.VAR_KEYWORD for p in params_iter):
        return params  # **kwargs: everything is accepted
    accepted = {p.name for p in params_iter if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    kept = {}
    for name, value in params.items():
        if name in accepted:
            kept[name] = value
        else:
            DROPPED_PARAMS.append(f"{key}: dropped {name!r} (not supported by this library version)")
    return kept


def _sk(name):
    def build(params):
        module_name, _, attr = name.rpartition(".")
        cls = getattr(__import__(f"sklearn.{module_name}", fromlist=[attr]), attr)
        return cls(**_filter_signature(name, cls, params))

    return build


def _xgb_reg(params):
    from xgboost import XGBRegressor

    params.setdefault("verbosity", 0)
    params.setdefault("tree_method", "hist")
    return XGBRegressor(**_filter_signature("xgboost_reg", XGBRegressor, params))


def _xgb_clf(params):
    from xgboost import XGBClassifier

    params.setdefault("verbosity", 0)
    params.setdefault("tree_method", "hist")
    return XGBClassifier(**_filter_signature("xgboost_clf", XGBClassifier, params))


def _lgbm_reg(params):
    from lightgbm import LGBMRegressor

    if "subsample" in params:
        params.setdefault("subsample_freq", 1)
    params.setdefault("verbose", -1)
    return LGBMRegressor(**_filter_signature("lightgbm_reg", LGBMRegressor, params))


def _lgbm_clf(params):
    from lightgbm import LGBMClassifier

    if "subsample" in params:
        params.setdefault("subsample_freq", 1)
    params.setdefault("verbose", -1)
    return LGBMClassifier(**_filter_signature("lightgbm_clf", LGBMClassifier, params))


def _torch_reg(params):
    from .dl.estimators import TorchMLPRegressor

    return TorchMLPRegressor(**_filter_signature("torch_mlp_reg", TorchMLPRegressor, params))


def _torch_clf(params):
    from .dl.estimators import TorchMLPClassifier

    return TorchMLPClassifier(**_filter_signature("torch_mlp_clf", TorchMLPClassifier, params))


def _torch_autoencoder(params):
    from .dl.estimators import TorchAutoencoder

    return TorchAutoencoder(**_filter_signature("torch_autoencoder", TorchAutoencoder, params))


_CONSTRUCTORS = {
    "linear": _sk("linear_model.LinearRegression"),
    "ridge": _sk("linear_model.Ridge"),
    "lasso": _sk("linear_model.Lasso"),
    "elasticnet": _sk("linear_model.ElasticNet"),
    "bayesian_ridge": _sk("linear_model.BayesianRidge"),
    "huber": _sk("linear_model.HuberRegressor"),
    "pls": _sk("cross_decomposition.PLSRegression"),
    "knn_reg": _sk("neighbors.KNeighborsRegressor"),
    "svr": _sk("svm.SVR"),
    "decision_tree_reg": _sk("tree.DecisionTreeRegressor"),
    "random_forest_reg": _sk("ensemble.RandomForestRegressor"),
    "extra_trees_reg": _sk("ensemble.ExtraTreesRegressor"),
    "gradient_boosting_reg": _sk("ensemble.GradientBoostingRegressor"),
    "hist_gb_reg": _sk("ensemble.HistGradientBoostingRegressor"),
    "adaboost_reg": _sk("ensemble.AdaBoostRegressor"),
    "logistic": _sk("linear_model.LogisticRegression"),
    "logistic_l1": _sk("linear_model.LogisticRegression"),
    "ridge_clf": _sk("linear_model.RidgeClassifier"),
    "linear_svc": _sk("svm.LinearSVC"),
    "knn_clf": _sk("neighbors.KNeighborsClassifier"),
    "svc": _sk("svm.SVC"),
    "decision_tree_clf": _sk("tree.DecisionTreeClassifier"),
    "random_forest_clf": _sk("ensemble.RandomForestClassifier"),
    "extra_trees_clf": _sk("ensemble.ExtraTreesClassifier"),
    "gradient_boosting_clf": _sk("ensemble.GradientBoostingClassifier"),
    "hist_gb_clf": _sk("ensemble.HistGradientBoostingClassifier"),
    "adaboost_clf": _sk("ensemble.AdaBoostClassifier"),
    "gaussian_nb": _sk("naive_bayes.GaussianNB"),
    "kmeans": _sk("cluster.KMeans"),
    "dbscan": _sk("cluster.DBSCAN"),
    "agglomerative": _sk("cluster.AgglomerativeClustering"),
    "pca": _sk("decomposition.PCA"),
    "tsne": _sk("manifold.TSNE"),
    "xgboost_reg": _xgb_reg,
    "xgboost_clf": _xgb_clf,
    "lightgbm_reg": _lgbm_reg,
    "lightgbm_clf": _lgbm_clf,
    "torch_mlp_reg": _torch_reg,
    "torch_mlp_clf": _torch_clf,
    "torch_autoencoder": _torch_autoencoder,
}

_CLASSIFIER_KEYS = {m.key for m in MODELS if "binary" in m.tasks or "multiclass" in m.tasks}


def build_model(key: str, params: dict | None = None):
    DROPPED_PARAMS.clear()
    return make_model(key, params)


def is_classifier_key(key: str) -> bool:
    return key in _CLASSIFIER_KEYS


def is_unsupervised_key(key: str) -> bool:
    tasks = set(get_model_spec(key).tasks)
    return "unsupervised" in tasks and not tasks & {"regression", "binary", "multiclass"}


def default_params(key: str) -> dict:
    return get_model_spec(key).default_params()


def suggested_grid(key: str) -> dict:
    return get_model_spec(key).suggested_grid()


def param_default_grid(key: str) -> dict:
    """Every param with a suggested value list, for the tuning screen."""
    spec = get_model_spec(key)
    out = {}
    for p in spec.params:
        if p.kind in ("int", "float", "str") and p.grid:
            out[p.name] = list(p.grid)
        elif p.kind == "choice" and p.choices:
            out[p.name] = list(p.choices)
    return out


def n_estimator_param(key: str) -> str | None:
    """Name of the 'number of trees' parameter, if the model has one."""
    spec = get_model_spec(key)
    for p in spec.params:
        if p.name in ("n_estimators", "max_iter"):
            return p.name
    return None


def describe_params(key: str, params: dict) -> str:
    bits = [f"{k}={v}" for k, v in (params or {}).items() if v is not None]
    return f"{get_model_spec(key).label}({', '.join(bits)})" if bits else get_model_spec(key).label
