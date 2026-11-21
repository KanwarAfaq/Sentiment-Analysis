#!/usr/bin/env python3
" Four-seed UCSA-20 experiment."
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

MODELS = {
    "mbert-cased": "bert-base-multilingual-cased",
    "mbert-uncased": "bert-base-multilingual-uncased",
    "distilbert": "distilbert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
}
SEEDS = (42, 169, 356, 789)
CONDITIONS = ("retained", "removed")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
NEGATION_MARKERS = {"نہیں", "نہ", "نھیں", "مت", "غیر", "نا"}
EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SPLIT_SEED = 42
RESAMPLES = 1000
STATISTICS_SEED = 2026
EXPECTED_COUNTS = {"train": 6118, "validation": 1530, "test": 1932}


def normalize(text: str) -> str:
    text = re.sub(r"<.*?>", "", str(text))
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    return text.replace("ۀ", "ہ").replace("ۂ", "ہ").replace("ؤ", "و").replace("ئ", "ی")


def remove_stopwords(text: str, words: set[str]) -> str:
    return " ".join(word for word in text.split() if word not in words)


def read_source(path: Path, prefix: str) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(path)
    if not {"Tweets", "label"}.issubset(source.columns):
        raise ValueError(f"{path} must contain Tweets and label columns")
    df = source[["Tweets", "label"]].dropna().copy()
    df = df[df.Tweets.astype(str).str.strip().ne("")].copy()
    if not df.label.isin([0, 1]).all():
        raise ValueError(f"Unexpected labels in {path}; expected 0 and 1")
    df["label"] = df.label.astype(int)
    df["row_id"] = [f"{prefix}-{i}" for i in df.index]
    df["raw_text"] = df.Tweets.astype(str)
    df["normalized"] = df.raw_text.map(normalize)
    df = df[df.normalized.str.strip().ne("")].copy()
    conflicts = df.groupby("normalized").label.nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{prefix} has contradictory labels for identical normalized reviews")
    duplicates = int(df.duplicated("normalized").sum())
    df = df.drop_duplicates("normalized", keep="first").reset_index(drop=True)
    return df, {"source_rows": len(source), "duplicate_rows_removed": duplicates,
                "blank_or_missing_rows_removed": len(source) - len(df) - duplicates}


def count_classes(df: pd.DataFrame) -> dict:
    return {"total": len(df), "positive": int((df.label == 1).sum()),
            "negative": int((df.label == 0).sum())}


def prepare_data(train_path: Path, test_path: Path, words: set[str]) -> tuple[dict, dict]:
    train, train_record = read_source(train_path, "train")
    test, test_record = read_source(test_path, "test")
    for frame in (train, test):
        frame["without_stopwords"] = frame.normalized.map(lambda x: remove_stopwords(x, words))
    overlaps = train.normalized.isin(set(test.normalized)) | train.without_stopwords.isin(set(test.without_stopwords))
    removed_ids = train.loc[overlaps, "row_id"].tolist()
    train = train.loc[~overlaps].reset_index(drop=True)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    fit_idx, val_idx = next(split.split(train, train.label, train.without_stopwords))
    data = {"train": train.iloc[fit_idx].reset_index(drop=True),
            "validation": train.iloc[val_idx].reset_index(drop=True), "test": test}
    for representation in ("normalized", "without_stopwords"):
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if set(data[left][representation]) & set(data[right][representation]):
                raise AssertionError(f"Overlap between {left} and {right} under {representation}")
    sizes = {k: count_classes(v) for k, v in data.items()}
    if {k: v["total"] for k, v in sizes.items()} != EXPECTED_COUNTS:
        raise ValueError(f"Dataset does not match the manuscript's fixed split: {sizes}")
    audit = {"train_source": train_record, "test_source": test_record,
             "excluded_train_test_overlap_ids": removed_ids,
             "excluded_train_test_overlap_count": len(removed_ids),
             "split": "first StratifiedGroupKFold fold, n_splits=5, random_state=42, groups=stop-word-removed normalized text",
             "class_counts": sizes, "stopword_entries_unique": len(words),
             "cross_split_duplicate_checks": "passed"}
    return data, audit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Reviews(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, column: str):
        encoded = tokenizer(frame[column].tolist(), add_special_tokens=True,
                            max_length=MAX_LENGTH, padding="max_length", truncation=True,
                            return_attention_mask=True)
        self.ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        self.masks = torch.as_tensor(encoded["attention_mask"], dtype=torch.long)
        self.labels = torch.as_tensor(frame.label.to_numpy(copy=True), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.ids[index], "attention_mask": self.masks[index],
                "labels": self.labels[index]}


def data_loader(data: Reviews, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def scores(y: np.ndarray, predicted: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="macro", zero_division=0)
    per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
        y, predicted, average=None, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_precision": float(precision), "macro_recall": float(recall),
            "macro_f1": float(f1),
            "negative_precision": float(per_precision[0]),
            "negative_recall": float(per_recall[0]), "negative_f1": float(per_f1[0]),
            "positive_precision": float(per_precision[1]),
            "positive_recall": float(per_recall[1]), "positive_f1": float(per_f1[1])}


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device):
    model.eval()
    truth, guesses, probabilities, losses = [], [], [], []
    for batch in loader:
        labels = batch["labels"].to(device)
        output = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device), labels=labels)
        logits = output.logits
        truth.extend(labels.cpu().numpy().tolist())
        guesses.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        probabilities.extend(logits.softmax(dim=-1)[:, 1].cpu().numpy().tolist())
        losses.append(float(output.loss.item()))
    return (np.asarray(truth), np.asarray(guesses), np.asarray(probabilities),
            float(np.mean(losses)))


def train_one(model_name: str, checkpoint: str, condition: str, seed: int,
              data: dict, encoded: dict, device: torch.device,
              out_dir: Path) -> dict:
    from transformers import AutoModelForSequenceClassification

    seed_everything(seed)
    started = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_batches = data_loader(encoded["train"], True, seed)
    val_batches = data_loader(encoded["validation"], False, seed)
    test_batches = data_loader(encoded["test"], False, seed)
    total_steps = EPOCHS * len(train_batches)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def linear_schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_schedule)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device))
            output.loss.backward()
            optimizer.step()
            scheduler.step()
        y_val, pred_val, _, _ = predict(model, val_batches, device)
        val_metrics = scores(y_val, pred_val)
        print(f"    epoch {epoch}/{EPOCHS}: validation macro-F1={val_metrics['macro_f1']:.4f}", flush=True)
    y, predictions, prob, test_loss = predict(model, test_batches, device)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    result = {"model": model_name, "checkpoint": checkpoint, "condition": condition,
              "seed": seed, "test_size": len(y), "test_loss": test_loss,
              "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
              "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
              "runtime_seconds": round(time.time() - started, 2), **scores(y, predictions)}
    original = data["test"]
    if not np.array_equal(y, original.label.to_numpy()):
        raise AssertionError("Prediction order does not match fixed test reviews")
    result_rows = original[["row_id", "raw_text", "label"]].copy()
    result_rows["predicted_label"] = predictions
    result_rows["positive_probability"] = prob
    result_rows["error_type"] = np.select(
        [(y == 0) & (predictions == 1), (y == 1) & (predictions == 0)],
        ["FP", "FN"], default="correct")
    result_rows["has_negation"] = [bool(set(s.split()) & NEGATION_MARKERS) for s in original.normalized]
    result_rows["word_count"] = original.normalized.map(lambda x: len(x.split()))
    result_rows.to_csv(out_dir / "predictions" /
                       f"{model_name}_{condition}_seed{seed}.csv", index=False, encoding="utf-8-sig")
    del model, optimizer, scheduler, train_batches, val_batches, test_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), group in df.groupby(["model", "condition"], sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            n = len(values)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            margin = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
            rows.append({"model": model, "condition": condition, "metric": metric,
                         "seeds": n, "mean": mean, "sd": sd,
                         "ci95_lower_t": mean - margin, "ci95_upper_t": mean + margin})
    return pd.DataFrame(rows)


def comparison_pairs():
    items = []
    for other in ("mbert-cased", "mbert-uncased", "distilbert"):
        items.append(("model_comparison", ("xlm-roberta", "retained"), (other, "retained")))
    for model in sorted(MODELS):
        items.append(("stopword_ablation", (model, "retained"), (model, "removed")))
    return items


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, float(p_values[i]) * (len(order) - rank)))
        adjusted[i] = previous
    return adjusted


def aligned_predictions(out_dir: Path, model: str, condition: str):
    labels, ids, pred = None, None, []
    for seed in SEEDS:
        file = out_dir / "predictions" / f"{model}_{condition}_seed{seed}.csv"
        df = pd.read_csv(file, usecols=["row_id", "label", "predicted_label"])
        if ids is None:
            ids = df.row_id.to_numpy()
            labels = df.label.to_numpy(dtype=np.int8)
        elif not np.array_equal(ids, df.row_id.to_numpy()) or not np.array_equal(labels, df.label.to_numpy()):
            raise AssertionError("Test review alignment failed")
        pred.append(df.predicted_label.to_numpy(dtype=np.int8))
    return ids, labels, np.stack(pred)


def macro_f1_across_runs(labels: np.ndarray, pred: np.ndarray) -> float:
    truth = labels[None, :]
    tp = np.sum((pred == 1) & (truth == 1), axis=1)
    tn = np.sum((pred == 0) & (truth == 0), axis=1)
    fp = np.sum((pred == 1) & (truth == 0), axis=1)
    fn = np.sum((pred == 0) & (truth == 1), axis=1)
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp), float), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros(len(tn), float), where=(2 * tn + fp + fn) > 0)
    return float(((f1_pos + f1_neg) / 2).mean())


def paired_statistics(out_dir: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(STATISTICS_SEED)
    full_seed_tests, prediction_tests = [], []
    for family, (a_model, a_condition), (b_model, b_condition) in comparison_pairs():
        a = df[(df.model == a_model) & (df.condition == a_condition)].sort_values("seed")
        b = df[(df.model == b_model) & (df.condition == b_condition)].sort_values("seed")
        if list(a.seed) != list(b.seed) or list(a.seed) != sorted(SEEDS):
            raise AssertionError("Missing paired seed results")
        differences = a.macro_f1.to_numpy() - b.macro_f1.to_numpy()
        sd = float(differences.std(ddof=1))
        mean = float(differences.mean())
        margin = float(stats.t.ppf(0.975, len(SEEDS)-1) * sd / math.sqrt(len(SEEDS)))
        raw_seed_p = float(stats.ttest_rel(a.macro_f1, b.macro_f1).pvalue) if sd > 0 else (1.0 if abs(mean) < 1e-12 else 0.0)
        name = f"{a_model}/{a_condition} minus {b_model}/{b_condition}"
        full_seed_tests.append({"family": family, "comparison": name, "paired_seeds": 4,
                                "mean_macro_f1_difference": mean, "t_ci95_lower": mean-margin,
                                "t_ci95_upper": mean+margin, "raw_p": raw_seed_p})
        a_ids, y, pred_a = aligned_predictions(out_dir, a_model, a_condition)
        b_ids, yy, pred_b = aligned_predictions(out_dir, b_model, b_condition)
        if not np.array_equal(a_ids, b_ids) or not np.array_equal(y, yy):
            raise AssertionError("Paired conditions have different test data")
        observed = macro_f1_across_runs(y, pred_a) - macro_f1_across_runs(y, pred_b)
        boot = np.empty(RESAMPLES)
        extreme = 0
        for i in range(RESAMPLES):
            idx = rng.integers(0, len(y), size=len(y))
            boot[i] = macro_f1_across_runs(y[idx], pred_a[:, idx]) - macro_f1_across_runs(y[idx], pred_b[:, idx])
            swap = rng.random(len(y)) < 0.5
            perm_a = np.where(swap, pred_b, pred_a)
            perm_b = np.where(swap, pred_a, pred_b)
            permutation_effect = macro_f1_across_runs(y, perm_a) - macro_f1_across_runs(y, perm_b)
            extreme += abs(permutation_effect) >= abs(observed) - 1e-12
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        prediction_tests.append({"family": family, "comparison": name,
                                 "macro_f1_difference": observed,
                                 "bootstrap_ci95_lower": float(ci_low), "bootstrap_ci95_upper": float(ci_high),
                                 "raw_randomization_p": (extreme + 1) / (RESAMPLES + 1),
                                 "bootstrap_resamples": RESAMPLES, "randomization_permutations": RESAMPLES})
    seed_df, pred_df = pd.DataFrame(full_seed_tests), pd.DataFrame(prediction_tests)
    for frame, source_col in ((seed_df, "raw_p"), (pred_df, "raw_randomization_p")):
        frame["holm_adjusted_p"] = np.nan
        for family, indices in frame.groupby("family").groups.items():
            frame.loc[indices, "holm_adjusted_p"] = holm_adjust(frame.loc[indices, source_col].to_numpy(dtype=float))
    return seed_df, pred_df


def persistent_errors(out_dir: Path) -> pd.DataFrame:
    cases = []
    for model in MODELS:
        for condition in CONDITIONS:
            frames = []
            for seed in SEEDS:
                path = out_dir / "predictions" / f"{model}_{condition}_seed{seed}.csv"
                df = pd.read_csv(path)
                df = df[["row_id", "raw_text", "label", "predicted_label", "error_type", "has_negation", "word_count"]].copy()
                df["seed"] = seed
                frames.append(df[df.error_type.isin(["FP", "FN"])])
            wrong = pd.concat(frames, ignore_index=True)
            for row_id, group in wrong.groupby("row_id"):
                if group.seed.nunique() != 4 or group.error_type.nunique() != 1:
                    continue
                first = group.iloc[0]
                cases.append({"model": model, "condition": condition, "row_id": row_id,
                              "reference_label": int(first.label), "error_type": first.error_type,
                              "predicted_label": int(first.predicted_label),
                              "seed42": int(group.loc[group.seed == 42, "predicted_label"].iloc[0]),
                              "seed169": int(group.loc[group.seed == 169, "predicted_label"].iloc[0]),
                              "seed356": int(group.loc[group.seed == 356, "predicted_label"].iloc[0]),
                              "seed789": int(group.loc[group.seed == 789, "predicted_label"].iloc[0]),
                              "has_negation": bool(first.has_negation),
                              "word_count": int(first.word_count), "raw_text": first.raw_text})
    return pd.DataFrame(cases, columns=["model", "condition", "row_id", "reference_label",
                                        "error_type", "predicted_label", "seed42", "seed169",
                                        "seed356", "seed789", "has_negation", "word_count", "raw_text"])


def parse_args():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=directory / "train_dataset.csv")
    parser.add_argument("--test-csv", type=Path, default=directory / "test_dataset.csv")
    parser.add_argument("--stopwords", type=Path, default=directory / "stopwords.txt")
    parser.add_argument("--out-dir", type=Path, default=directory / "outputs")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run():
    from transformers import AutoTokenizer

    args = parse_args()
    paths = {"train_csv": args.train_csv.resolve(), "test_csv": args.test_csv.resolve(),
             "stopwords": args.stopwords.resolve()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    with args.stopwords.open(encoding="utf-8") as stream:
        words = {line.strip() for line in stream if line.strip()}
    data, audit = prepare_data(args.train_csv, args.test_csv, words)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    metrics_file = out_dir / "run_metrics.csv"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output folder is not empty: {out_dir}. Use --resume or choose another --out-dir.")
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    all_results = pd.read_csv(metrics_file).to_dict("records") if args.resume and metrics_file.exists() else []
    completed = {(r["model"], r["condition"], int(r["seed"])) for r in all_results}
    if len(completed) != len(all_results):
        raise ValueError("Repeated run configurations in saved metrics")
    print(f"Fixed data split: {audit['class_counts']} | Device: {device}", flush=True)
    for model_name, checkpoint in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        for condition in CONDITIONS:
            column = "normalized" if condition == "retained" else "without_stopwords"
            encoded = {name: Reviews(frame, tokenizer, column) for name, frame in data.items()}
            for seed in SEEDS:
                slug = f"{model_name}_{condition}_seed{seed}"
                if (model_name, condition, seed) in completed:
                    if not (out_dir / "predictions" / f"{slug}.csv").is_file():
                        raise ValueError(f"Cannot resume: prediction file missing for {slug}")
                    print(f"  Already completed: {slug}", flush=True)
                    continue
                print(f"  Training {slug}", flush=True)
                row = train_one(model_name, checkpoint, condition, seed, data,
                                encoded, device, out_dir)
                all_results.append(row)
                pd.DataFrame(all_results).to_csv(metrics_file, index=False)
                completed.add((model_name, condition, seed))
            del encoded
            gc.collect()
        del tokenizer
    if len(completed) != 32:
        raise AssertionError("The revised experiment requires all 32 model/condition/seed runs")
    result_df = pd.DataFrame(all_results)
    if not set(zip(result_df.model, result_df.condition, result_df.seed.astype(int))) == {
        (m, c, s) for m in MODELS for c in CONDITIONS for s in SEEDS
    }:
        raise AssertionError("Saved run configurations are incomplete")
    seed_summary(result_df).to_csv(out_dir / "metric_summary.csv", index=False)
    seed_tests, prediction_tests = paired_statistics(out_dir, result_df)
    seed_tests.to_csv(out_dir / "paired_seed_tests.csv", index=False)
    prediction_tests.to_csv(out_dir / "paired_prediction_tests.csv", index=False)
    persistent_errors(out_dir).to_csv(out_dir / "persistent_error_cases.csv", index=False, encoding="utf-8-sig")
    print("Completed all 32 real-model runs; manuscript values require comparison with the archived results.", flush=True)


if __name__ == "__main__":
    run()