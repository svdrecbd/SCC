"""Compare fixed inference and fitted public learners on predeclared fresh tasks."""
import json
import os
from pathlib import Path
import sys
import time

from validate_interface import parameter_digest


def generate_task(configuration, family, seed):
    import numpy as np
    from scipy.special import softmax

    generator = np.random.default_rng(seed)
    count = configuration["context_count"] + configuration["query_count"]
    features = configuration["feature_count"]
    classes = configuration["class_count"]
    inputs = generator.normal(size=(count, features))
    if family == "linear":
        coefficients = generator.normal(size=(features, classes)) / np.sqrt(features)
        probability = softmax(2 * inputs @ coefficients, axis=1)
    elif family == "smooth_interactions":
        hidden = np.tanh(inputs @ generator.normal(size=(features, 16)) / np.sqrt(features))
        probability = softmax(2 * hidden @ generator.normal(size=(16, classes)) / 4, axis=1)
    elif family == "axis_partitions":
        partition = (inputs[:, :4] > 0) @ (2 ** np.arange(4))
        table = softmax(2 * generator.normal(size=(16, classes)), axis=1)
        probability = table[partition]
    elif family == "gaussian_classes":
        means = generator.normal(size=(classes, features)) * 0.8
        latent_class = generator.integers(classes, size=count)
        inputs += means[latent_class]
        distances = ((inputs[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
        probability = softmax(-distances / 2, axis=1)
    else:
        raise ValueError(family)
    outcomes = (generator.random(count)[:, None] > np.cumsum(probability, axis=1)).sum(axis=1)
    context_count = configuration["context_count"]
    assert set(outcomes[:context_count]) == set(range(classes)), "Do not replace a failed task seed"
    return inputs[:context_count], outcomes[:context_count], inputs[context_count:], outcomes[context_count:], probability[context_count:]


def public_models(seed):
    from lightgbm import LGBMClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {}
    for regularization in (0.1, 1, 10):
        models[f"logistic_{regularization}"] = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=1000))
    for regularization in (1, 10):
        models[f"radial_svm_{regularization}"] = make_pipeline(StandardScaler(), SVC(C=regularization, probability=True, random_state=seed))
    for leaf_size in (1, 5):
        models[f"random_forest_{leaf_size}"] = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf_size, random_state=seed, n_jobs=1)
        models[f"extra_trees_{leaf_size}"] = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=leaf_size, random_state=seed, n_jobs=1)
    models["histogram_boosting"] = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1, random_state=seed)
    models["light_gradient_boosting"] = LGBMClassifier(n_estimators=100, num_leaves=7, min_child_samples=10, n_jobs=1, verbosity=-1, random_state=seed)
    for neighbors in (5, 15, 31):
        models[f"nearest_neighbors_{neighbors}"] = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbors, weights="distance", n_jobs=1))
    models["linear_discriminant"] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    models["quadratic_discriminant"] = QuadraticDiscriminantAnalysis(reg_param=0.1)
    models["gaussian_naive_bayes"] = GaussianNB()
    return models


def main(directory, family, seed):
    started = time.monotonic()
    import numpy as np
    import torch
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration = json.loads((directory / "config.json").read_text())
    assert family in configuration["families"] and seed in configuration["seeds"]
    output = directory / f"{family}_{seed}"
    output.mkdir(exist_ok=False)
    context, labels, queries, outcomes, truth = generate_task(configuration, family, seed)
    np.savez_compressed(output / "task.npz", context=context, labels=labels, queries=queries, outcomes=outcomes, true_probability=truth)
    classes = configuration["class_count"]
    target = np.eye(classes)[outcomes]
    context_target = np.eye(classes)[labels]
    predictions = {}
    timing = {}
    validation_losses = {}
    models = public_models(seed)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(context, labels))
    public_started = time.monotonic()
    validation_predictions = {}
    for name, model in models.items():
        model_started = time.monotonic()
        validation_probability = np.zeros_like(context_target)
        for fitting, validation in folds:
            fitted = clone(model).fit(context[fitting], labels[fitting])
            validation_probability[validation] = fitted.predict_proba(context[validation])
        validation_predictions[name] = validation_probability
        validation_losses[name] = float(np.mean(np.sum((validation_probability - context_target) ** 2, axis=1)))
        fitted = clone(model).fit(context, labels)
        predictions[name] = fitted.predict_proba(queries)
        timing[name] = time.monotonic() - model_started
    predictions["public_average"] = np.mean(list(predictions.values()), axis=0)
    validation_losses["public_average"] = float(np.mean(np.sum((np.mean(list(validation_predictions.values()), axis=0)-context_target)**2, axis=1)))
    selected = min(validation_losses, key=validation_losses.get)
    predictions["public_selected"] = predictions[selected]
    public_seconds = time.monotonic() - public_started
    predictions["context_frequency"] = np.tile(np.bincount(labels, minlength=classes) / len(labels), (len(queries), 1))
    predictions["known_law"] = truth

    model_started = time.monotonic()
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, model_path=configuration["checkpoint"], device="cpu",
        n_estimators=configuration["ensemble_count"], random_state=seed,
        n_preprocessing_jobs=1, inference_precision=torch.float32,
        tuning_config=None, show_progress_bar=False)
    model.fit(context, labels)
    initial_digest, parameters = parameter_digest(model.models_)
    predictions["contextual_model"] = model.predict_proba(queries)
    inference_seconds = time.monotonic() - model_started
    assert parameter_digest(model.models_)[0] == initial_digest

    baseline = predictions["public_selected"]
    useful = predictions["contextual_model"]
    baseline_risk = 1 - np.cumsum(baseline, axis=1)[:, :-1]
    displacement = useful - baseline
    recovered = np.clip(baseline_risk + np.diff(displacement, axis=1) / 4, 0, 1)
    indicator = outcomes[:, None] > np.arange(classes - 1)[None, :]
    useful_gain = ((baseline - target)**2).sum(axis=1) - ((useful - target)**2).sum(axis=1)
    risk_gain = ((baseline_risk - indicator)**2).sum(axis=1) - ((recovered - indicator)**2).sum(axis=1)
    assert np.all(risk_gain >= useful_gain / 4 - 1e-6)
    scores = {}
    for name, probability in predictions.items():
        assert probability.shape == target.shape and np.all(np.isfinite(probability))
        assert np.all(probability >= 0) and np.allclose(probability.sum(axis=1), 1, atol=1e-6)
        scores[name] = float(np.mean(((probability - target)**2).sum(axis=1)))
    np.savez_compressed(output / "predictions.npz", **predictions, recovered_risk=recovered, useful_gain=useful_gain, risk_gain=risk_gain)
    result = {"family": family, "seed": seed, "scores": scores, "public_selected": selected,
              "validation_losses": validation_losses, "public_model_seconds": timing,
              "public_portfolio_seconds": public_seconds, "contextual_model_seconds": inference_seconds,
              "parameter_count": parameters, "parameter_state_sha256": initial_digest,
              "mean_useful_gain": float(useful_gain.mean()), "mean_summed_risk_gain": float(risk_gain.mean()),
              "minimum_recovery_margin": float(np.min(risk_gain - useful_gain / 4)),
              "elapsed_seconds": time.monotonic() - started, "neural_training": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("scores", "validation_losses", "public_model_seconds")} ), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]))
