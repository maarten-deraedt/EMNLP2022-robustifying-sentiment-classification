from typing import List, Dict

from models import MeanOffset, RandomOffset, MeanOffsetRegressor, LinearRegressor, OriginalMeanOffset
from sklearn.linear_model import LogisticRegressionCV

import numpy as np


class Evaluable(object):
    def __init__(self, max_iter: int, cv: int, embedder_name: str, model_name: str):
        """
        :param max_iter: maximum number of iterations for the LogisticRegressionCV to converge.
        :param cv: the number of folds of the LogitisticRegressionCV model.
        :param embedder_name: the sentence embedder name, used for loading the correct test and training sets.
        """
        self.max_iter = max_iter
        self.cv = cv
        self.embedder_name = embedder_name
        self.model_name = model_name

        """Load the OOD test sets for the given sentence embedder."""
        self.X_amazon, self.y_amazon = (
            np.load(
                f"encodings/{embedder_name}/X_amazon.npy"
            ),
            np.load(
                f"encodings/{embedder_name}/y_amazon.npy"
            ),
        )
        self.X_semeval, self.y_semeval = (
            np.load(
                f"encodings/{embedder_name}/X_semeval.npy"
            ),
            np.load(
                f"encodings/{embedder_name}/y_semeval.npy"
            ),
        )
        self.X_yelp, self.y_yelp = (
            np.load(
                f"encodings/{embedder_name}/X_yelp.npy"
            ),
            np.load(
                f"encodings/{embedder_name}/y_yelp.npy"
            ),
        )
        """Load the ID and CAD (ctf) test sets of IMDb for the given sentence embedder."""
        self.X_test, self.y_test = (
            np.load(f"encodings/{embedder_name}/X_test.npy"),
            np.load(f"encodings/{embedder_name}/y_test.npy"),
        )
        self.X_test_ctf, self.y_test_ctf = (
            np.load(f"encodings/{embedder_name}/X_test_ctf.npy"),
            np.load(f"encodings/{embedder_name}/y_test_ctf.npy"),
        )

        """
        Construct the lists for storing the metrics of the different runs for a fixed value of K.
        The metrics computed are the accuracies 5 test sets.
        """
        self.orig_accs = []
        self.ctf_accs = []
        self.amazon_accs = []
        self.semeval_accs = []
        self.yelp_accs = []

        """The specific features for training for a fixed combination of K and seed (run). These are set by calling the prepare_features method."""
        self.X, self.y, self.sample_weight = None, None, None

    def reset_(self) -> None:
        """Reset the calculated metrics after each run."""
        self.orig_accs = []
        self.ctf_accs = []
        self.amazon_accs = []
        self.semeval_accs = []
        self.yelp_accs = []

        self.X, self.y, self.sample_weight = None, None, None

    def prepare_features_(self, features: Dict, n_pair: int) -> None:
        """To be implemented by the subclasses."""
        pass

    def evaluate(self, features: Dict, seed: int, lambdas: List[float], n_pair: int) -> None:
        """
        Evaluate this model for a single run (seed) and single K.
        Before running evaluate, the prepare_features call ensures that the correct training data is stored in self.X and self.y.
        :param lambdas: strong, or free regularization parameters.
        :param sample_weight: downscaling factor for the original samples, only used for the weighted baseline.
        :return:
        """
        self.prepare_features_(features, n_pair)
        if self.sample_weight is not None:  # only used for the weighted baseline.
            model = LogisticRegressionCV(
                max_iter=self.max_iter,
                random_state=seed,
                cv=self.cv,
                n_jobs=-1,
                refit=True,
                Cs=lambdas,
            ).fit(self.X, self.y, sample_weight=self.sample_weight)
        else:
            model = LogisticRegressionCV(
                max_iter=self.max_iter,
                random_state=seed,
                cv=self.cv,
                n_jobs=-1,
                refit=True,
                Cs=lambdas,
            ).fit(self.X, self.y)

        self.orig_accs.append(float(model.score(self.X_test, self.y_test)))
        self.ctf_accs.append(float(model.score(self.X_test_ctf, self.y_test_ctf)))
        self.amazon_accs.append(float(model.score(self.X_amazon, self.y_amazon)))
        self.semeval_accs.append(float(model.score(self.X_semeval, self.y_semeval)))
        self.yelp_accs.append(float(model.score(self.X_yelp, self.y_yelp)))

    def set_results(
        self, results: Dict, n_pair: int,
    ):
        results[self.model_name][f"run={n_pair}"] = {
            "mean_ctf": np.mean(self.ctf_accs),
            "mean_orig": np.mean(self.orig_accs),
            "std_ctf": np.std(self.ctf_accs),
            "std_orig": np.std(self.orig_accs),
            "mean_amazon": np.mean(self.amazon_accs),
            "std_amazon": np.std(self.amazon_accs),
            "mean_semeval": np.mean(self.semeval_accs),
            "std_semeval": np.std(self.semeval_accs),
            "mean_yelp": np.mean(self.yelp_accs),
            "std_yelp": np.std(self.yelp_accs),
            "orig_accs": self.orig_accs,
            "ctf_accs": self.ctf_accs,
            "ood_accs": [(t[0] + t[1] + t[2])/3 for t in zip(self.amazon_accs, self.semeval_accs, self.yelp_accs)],
        }
        self.reset_()  # empty the metric lists to prepare for the 50 runs of the next K.

    def summary_print(self, results: Dict, n_pair: int) -> None:
        """
        Print aggregated metrics after completing all the runs for a given K(=n_pair).
        :param results: Global dictionary to which all the metrics are written for each K (aggregated across the 50 runs).
        :param n_pair: K
        :return:
        """
        mean_orig = results[self.model_name][f"run={n_pair}"]["mean_orig"]
        mean_ctf = results[self.model_name][f"run={n_pair}"]["mean_ctf"]
        mean_amazon = results[self.model_name][f"run={n_pair}"]["mean_amazon"]
        mean_semeval = results[self.model_name][f"run={n_pair}"]["mean_semeval"]
        mean_yelp = results[self.model_name][f"run={n_pair}"]["mean_yelp"]

        average_ood = np.mean([mean_amazon, mean_semeval, mean_yelp])
        average = np.mean([mean_orig, mean_ctf, average_ood])
        print(
            f"{self.model_name}:\t\t\t\t\t\tAVG.={average:.3f} | ID={mean_orig:.3f} |  CAD={mean_ctf:.3f} | OOD={average_ood:.3f}"
        )


class Paired(Evaluable):
    def __init__(self, max_iter: int, cv: int, embedder_name: str, model_name: str):
        super().__init__(max_iter, cv, embedder_name, model_name)

    def prepare_features_(self, features: Dict, n_pair: int) -> None:
        self.X = features["paired"]["X"]
        self.y = features["paired"]["y"]


class Weighted(Evaluable):
    def __init__(self, max_iter: int, cv: int, embedder_name: str, model_name: str):
        super().__init__(max_iter, cv, embedder_name, model_name)

    def prepare_features_(self, features: Dict, n_pair: int) -> None:
        self.X = np.concatenate(
            (features["original"]["X"], features["sampled_ctf"]["X"])
        )
        self.y = np.concatenate(
            (features["original"]["y"], features["sampled_ctf"]["y"])
        )
        original_weight = n_pair / features["original"]["X"].shape[0]
        self.sample_weight = np.ones(shape=(self.y.size))
        self.sample_weight[:features["original"]["X"].shape[0]] = original_weight


class ExtraOriginal(Evaluable):
    def __init__(self, max_iter: int, cv: int, embedder_name: str, model_name: str):
        super().__init__(max_iter, cv, embedder_name, model_name)

    def prepare_features_(self, features: Dict, n_pair: int) -> None:
        self.X = np.concatenate(
            (features["original"]["X"], features["sampled_orig_extra"]["X"])
        )
        self.y = np.concatenate(
            (features["original"]["y"], features["sampled_orig_extra"]["y"])
        )


class Augmentor(Evaluable):
    def __init__(
        self,
        max_iter: int,
        cv: int,
        embedder_name: str,
        model_name: str,
        augmentor_type: str,
    ):
        super().__init__(max_iter, cv, embedder_name, model_name)
        self.augmentor_type = augmentor_type

    def prepare_features_(self, features: Dict, n_pair: int) -> None:
        X_original, y_original = (
            features["original"]["X"],
            features["original"]["y"],
        )
        X_original_pos = X_original[y_original == 1]
        X_original_neg = X_original[y_original == 0]

        if self.augmentor_type == "mean_offset":
            posneg = MeanOffset(
                X=features["pos_neg_ctf"]["X"],
                y=features["pos_neg_ctf"]["y"],
                src_label=1,
            )
            negpos = MeanOffset(
                X=features["neg_pos_ctf"]["X"],
                y=features["neg_pos_ctf"]["y"],
                src_label=0,
            )
        elif self.augmentor_type == "original_mean_offset":
            posneg = OriginalMeanOffset(
                X_orig=features["original"]["X"],
                y_orig=features["original"]["y"],
                src_label=1,
            )
            negpos = OriginalMeanOffset(
                X_orig=features["original"]["X"],
                y_orig=features["original"]["y"],
                src_label=0,
            )
        elif self.augmentor_type == "random_offset":
            posneg = RandomOffset(
                X=features["pos_neg_ctf"]["X"],
                y=features["pos_neg_ctf"]["y"],
                src_label=1,
            )
            negpos = RandomOffset(
                X=features["neg_pos_ctf"]["X"],
                y=features["neg_pos_ctf"]["y"],
                src_label=0,
            )
        elif self.augmentor_type == "mean_offset_residual":
            posneg = MeanOffsetRegressor(
                X=features["pos_neg_ctf"]["X"],
                y=features["pos_neg_ctf"]["y"],
                src_label=1,
            )
            negpos = MeanOffsetRegressor(
                X=features["neg_pos_ctf"]["X"],
                y=features["neg_pos_ctf"]["y"],
                src_label=0,
            )
        elif self.augmentor_type == "linear_regression":
            posneg = LinearRegressor(
                X=features["pos_neg_ctf"]["X"],
                y=features["pos_neg_ctf"]["y"],
                src_label=1,
            )
            negpos = LinearRegressor(
                X=features["neg_pos_ctf"]["X"],
                y=features["neg_pos_ctf"]["y"],
                src_label=0,
            )

        X_pos_ctf, y_pos_ctf = negpos.transform(
            X=X_original_neg, y=y_original[y_original == 0]
        )
        X_neg_ctf, y_neg_ctf = posneg.transform(
            X=X_original_pos, y=y_original[y_original == 1]
        )
        self.X = np.concatenate((X_original_neg, X_pos_ctf, X_original_pos, X_neg_ctf))
        self.y = np.concatenate((y_original[y_original == 0], y_pos_ctf, y_original[y_original == 1], y_neg_ctf))


class BaselineAAAI(Evaluable):
    """
    AAAI 2021, Robustness to Spurious Correlations in Text Classification via Automatically Generated Counterfactuals
    """
    def __init__(self, max_iter: int, cv: int, embedder_name: str, model_name: str, ctf_type: str = ""):
        super().__init__(max_iter, cv, embedder_name, model_name)
        self.ctf_type = ctf_type

    def prepare_features_(self, features: Dict, n_pair: int) -> None:
        self.X = np.load(f"encodings/{self.embedder_name}/X_auto_train_original+{self.ctf_type}.npy")
        self.y = np.load(f"encodings/{self.embedder_name}/y_auto_train_original+{self.ctf_type}.npy")
