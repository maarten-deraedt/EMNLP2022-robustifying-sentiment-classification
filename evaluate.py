from featurizers import imdb_features
from evaluators import (
    ExtraOriginal,
    Paired,
    Weighted,
    Augmentor,
    BaselineAAAI,
)

import time
import json
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    SBERT: "all-roberta-large-v1", "all-distilroberta-v1", "all-mpnet-base-v2"
    SimCSE: "unsup-simcse-roberta-large", "unsup-simcse-bert-large-uncased", "unsup-simcse-bert-base-uncased"
    """
    parser.add_argument("-n", "--name", type=str)
    args = parser.parse_args()
    embedder_name = args.name

    print(f"Starting to run the experiments for {embedder_name}.")

    """Write the values of the evaluation to out_file."""
    out_file = f"results/metrics/{embedder_name}.json"

    n_pairs = [16, 32, 64, 128]  # K values, i.e. how many counterfactuals are sampled.
    n_runs = 50         # Number of randomly initialized runs for each value of K.
    n_runs_base = 1     # Number of runs for baselines models (these baselines don't rely on randomly sampling k counterfactuals or n extra originals).
    cv = 4              # Number of folds for cross-validation to determine optimal inverse regularization strength C.
    max_iter = 4000     # Maximum of number of iterations of the logistic regression classifier before convergence.

    # Choose random seed between lower and upper for each run.
    random_lower, random_upper = 99999, 9999999

    # Free regularization values for L2 logistic regression.
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Strong regularization values for L2 logistic regression (used for the classifiers trained on generated counterfactuals)
    lambdas_strong = [0.001, 0.01, 0.1, 1]

    """Initialize the dictionary in which the metrics are stored."""
    results = {
        "orig_extra": dict(),
        "orig_extra_strong": dict(),
        "paired": dict(),
        "paired_strong": dict(),
        "weighted": dict(),
        "weighted_strong": dict(),
        "mean_offset": dict(),
        "mean_offset_strong": dict(),
        "mean_offset_residual": dict(),
        "mean_offset_residual_strong": dict(),
        "ctf_by_predicted_causal": dict(),
        "ctf_by_predicted_causal_strong": dict(),
        "ctf_by_annotated_causal": dict(),
        "ctf_by_annotated_causal_strong": dict(),
        "ctf_by_all_causal": dict(),
        "ctf_by_all_causal_strong": dict(),
        "random_offset": dict(),
        "random_offset_strong": dict(),
        "original_mean_offset": dict(),
        "original_mean_offset_strong": dict(),
        "linear_regression": dict(),
        "linear_regression_strong": dict(),
    }
    """The base models don't rely on sampling k counterfactuals or sampling extra original samples, and thus do not exhibit variance across different runs. 
    The baselines of AAAI 2021, for which the original 1.7k id samples are automatically augmented in the input space with increasing levels
    of human annotation.
    """
    base_models = [
        BaselineAAAI(max_iter, cv, embedder_name, "ctf_by_predicted_causal", "ctf_by_predicted_causal"),    # Predicted from top. (n'' = 1284)
        BaselineAAAI(max_iter, cv, embedder_name, "ctf_by_annotated_causal", "ctf_by_annotated_causal"),    # Annotated from top. (n'' = 1618)
        BaselineAAAI(max_iter, cv, embedder_name, "ctf_by_all_causal", "ctf_by_all_causal"),                # Annotated from all. (n'' = 1694)
    ]
    """Run the base models with only the strong L2 regularization values."""
    base_models_strong = [
        BaselineAAAI(max_iter, cv, embedder_name, "ctf_by_predicted_causal_strong", "ctf_by_predicted_causal"),
        BaselineAAAI(max_iter, cv, embedder_name, "ctf_by_annotated_causal_strong", "ctf_by_annotated_causal"),
        BaselineAAAI(max_iter, cv, embedder_name, "ctf_by_all_causal_strong", "ctf_by_all_causal"),
    ]
    base_models_all = base_models + base_models_strong
    for i in range(n_runs_base):
        features = imdb_features(encoding_name=embedder_name, n_pairs=0, n_extra_orig=0)
        seed = random.randint(random_lower, random_upper)
        for base_model in base_models:
            base_model.evaluate(features, seed, lambdas, 0)
        for base_model_strong in base_models_strong:
            base_model_strong.evaluate(features, seed, lambdas_strong, 0)
    for model in base_models_all:
        model.set_results(results, 0)
        model.summary_print(results, 0)

    """
    Freely regularized classifiers.
    """
    models = [
        ExtraOriginal(max_iter, cv, embedder_name, "orig_extra"), # The original baseline presented in the main results: 1,707 extra sampled originals.
        Paired(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="paired"),
        Weighted(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="weighted"),
        Augmentor(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="mean_offset", augmentor_type="mean_offset"),
        Augmentor(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="mean_offset_residual", augmentor_type="mean_offset_residual"),
    ]
    """
    Strongly regularized classifiers.
    """
    models_strong = [
        ExtraOriginal(max_iter, cv, embedder_name, "orig_extra_strong"),
        Paired(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="paired_strong"),
        Weighted(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="weighted_strong"),
        Augmentor(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="mean_offset_strong", augmentor_type="mean_offset"),
        Augmentor(max_iter=max_iter, cv=cv, embedder_name=embedder_name, model_name="mean_offset_residual_strong", augmentor_type="mean_offset_residual"),
    ]
    all_models = models + models_strong
    for n_pair in n_pairs:
        print(f"K={n_pair}")
        start = time.time()
        for i in range(n_runs):
            seed = random.randint(random_lower, random_upper)
            # A subset of k (k/2 (pos x, neg xcad) and k/2 (neg x, pos xcad)) counterfactuals are sampled in each run.
            features = imdb_features(encoding_name=embedder_name, n_pairs=n_pair, n_extra_orig=1707) # sample k(=n_pair) counterfactuals and 1,707(=n) extra originals.
            for model in models:
                model.evaluate(
                    features=features, seed=seed, lambdas=lambdas, n_pair=n_pair
                )
            for model_strong in models_strong:
                model_strong.evaluate(
                    features=features, seed=seed, lambdas=lambdas_strong, n_pair=n_pair
                )

        for model in all_models:
            model.set_results(results, n_pair)
            model.summary_print(results, n_pair)
        print(f"Took {time.time() - start} seconds to complete {n_runs} for K={n_pair}")

    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=4)