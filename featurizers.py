import numpy as np


def imdb_features(encoding_name, n_pairs: int, n_extra_orig: int = 0):
    X_original = np.load(f"encodings/{encoding_name}/X_original.npy")
    X_ctf = np.load(f"encodings/{encoding_name}/X_ctf.npy")
    y_original = np.load(f"encodings/{encoding_name}/y_original.npy")
    y_ctf = np.load(f"encodings/{encoding_name}/y_ctf.npy")
    X_extra = np.load(f"encodings/{encoding_name}/X_extra.npy")
    y_extra = np.load(f"encodings/{encoding_name}/y_extra.npy")

    """
    Shuffle the data
    """
    p = np.random.permutation(X_original.shape[0])
    X_original, y_original, X_ctf, y_ctf = (
        X_original[p],
        y_original[p],
        X_ctf[p],
        y_ctf[p],
    )
    X_pos_all, X_neg_all, X_neg_ctf_all, X_pos_ctf_all = (
        X_original[y_original == 1],
        X_original[y_original == 0],
        X_ctf[y_ctf == 0],
        X_ctf[y_ctf == 1],
    )

    p_extra = np.random.permutation(X_extra.shape[0])
    X_extra, y_extra = X_extra[p_extra], y_extra[p_extra]
    X_sampled_orig_extra, y_sampled_orig_extra = X_extra[:n_extra_orig, :], y_extra[:n_extra_orig]

    """
    Features for paired baseline
    """
    n_per_class = int(n_pairs / 2)
    X_pos = X_pos_all[:n_per_class, :]
    X_neg_ctf = X_neg_ctf_all[:n_per_class, :]
    X_pos_neg_ctf = np.concatenate((X_pos, X_neg_ctf))
    y_pos_neg = np.ones(shape=(X_pos_neg_ctf.shape[0],), dtype=np.int64)
    y_pos_neg[X_pos.shape[0]:] = np.int64(0)

    X_neg = X_neg_all[:n_per_class, :]
    X_pos_ctf = X_pos_ctf_all[:n_per_class, :]
    X_neg_pos_ctf = np.concatenate((X_neg, X_pos_ctf))
    y_neg_pos = np.zeros(shape=(X_neg_pos_ctf.shape[0],), dtype=np.int64)
    y_neg_pos[X_neg.shape[0]:] = np.int64(1)

    X_paired = np.concatenate((X_pos_neg_ctf, X_neg_pos_ctf))
    y_paired = np.concatenate((y_pos_neg, y_neg_pos))
    """"""

    """
    Features to augment all original sentences that do not have a counterfactual version
    """
    X_pos_no_ctf = X_pos_all[n_per_class:, :]
    X_neg_no_ctf = X_neg_all[n_per_class:, :]
    X_no_ctf = np.concatenate((X_pos_no_ctf, X_neg_no_ctf))
    y_no_ctf = np.zeros(shape=(X_no_ctf.shape[0],))
    y_no_ctf[:X_pos_no_ctf.shape[0]] = 1
    """"""

    y_sampled_orig = np.zeros(shape=(n_pairs,), dtype=np.int64)
    y_sampled_orig[n_per_class:] = 1

    """
    Sampled CTF for weighted baseline: train on ALL original data, equally weighted (i.e., downscaled by factor k/n) with the available CTF pairs
    """
    X_sampled_ctf = np.concatenate((X_pos_ctf, X_neg_ctf))
    y_sampled_ctf = np.zeros(shape=(X_sampled_ctf.shape[0],))
    y_sampled_ctf[:X_pos_ctf.shape[0]] = np.int64(1)

    features = {
        "original": {"X": X_original, "y": y_original,},            # The original 1,707 samples.
        "pos_neg_ctf": {"X": X_pos_neg_ctf, "y": y_pos_neg},        # For the augmentor baselines (k/2 (positive x, negative xcad)).
        "neg_pos_ctf": {"X": X_neg_pos_ctf, "y": y_neg_pos},        # For the augmentor baselines (k/2 (negative x, positive xcad)).
        "paired": {"X": X_paired, "y": y_paired,},                  # For the Paired baseline.
        "sampled_ctf": {"X": X_sampled_ctf, "y": y_sampled_ctf,},   # The k sampled (k/2 (pos x, neg xcad) and k/2 (neg x, pos xcad) samples for the Weighted baseline.
        "original_no_ctf": {"X": X_no_ctf, "y": y_no_ctf},          # All the original samples without corresponding counterfactual: used for the Weighted baseline.
        "sampled_orig_extra": {"X": X_sampled_orig_extra, "y": y_sampled_orig_extra},   # The extra (randomly) picked original 1,707 samples.
    }

    return features