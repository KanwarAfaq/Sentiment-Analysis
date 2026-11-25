# A Controlled Four-Way Transformer Benchmark for Binary Urdu Sentiment Classification on UCSA-20

This repository contains the training and evaluation script (`revised_experiment.py`) for reproducing the four-model, four-seed binary Urdu sentiment classification experiments on the UCSA-20 corpus.

## Experimental Protocol

The script evaluates four pretrained multilingual encoders under a fixed training configuration:

* `bert-base-multilingual-cased` (mBERT-cased)
* `bert-base-multilingual-uncased` (mBERT-uncased)
* `distilbert-base-multilingual-cased` (DistilBERT)
* `xlm-roberta-base` (XLM-RoBERTa-base)

Each model is trained under two preprocessing conditions (`retained` and `removed` stop words) across four random seeds (`42`, `169`, `356`, `789`), yielding 32 runs in total.

### Data Partitioning and Preprocessing

1. Missing or blank reviews are removed, HTML tags and specified Urdu diacritics are stripped, and selected Urdu character variants are normalized.
2. Duplicate normalized reviews within each source file are removed, and label consistency is verified.
3. Training rows overlapping the held-out test set under either normalized text or stop-word-removed text are excluded.
4. The remaining training records are split using the first fold of a 5-fold `StratifiedGroupKFold` (`random_state=42`), grouped by stop-word-removed normalized text.
5. The resulting partition sizes are verified against the fixed counts reported in the manuscript: **6,118 training**, **1,530 validation**, and **1,932 test** reviews.

### Training Hyperparameters

* Epochs: 5 (evaluation is performed on the final epoch)
* Optimizer: AdamW (learning rate `1e-5`, weight decay `0.01`)
* Scheduler: Linear decay with 10% warmup (`warmup_ratio=0.1`)
* Batch size: 16
* Maximum sequence length: 128 tokens

### Statistical Evaluation

After all 32 runs complete, the script computes:

* Four-seed mean, sample standard deviation, and 95% Student's $t$ confidence intervals for accuracy, macro precision, macro recall, and macro-F1.
* Paired model comparisons (XLM-RoBERTa-base against each alternative, stop words retained) and stop-word ablations (retained minus removed per model).
* Seed-level paired $t$-tests and review-level paired bootstrap confidence intervals (1,000 resamples) with paired randomization tests (1,000 permutations), using Holm adjustment within each comparison family.
* Persistent error cases misclassified identically across all four seeds.

## Requirements
To ensure strict mathematical reproducibility (as reported in the manuscript), install the exact locked dependencies:
pip install -r requirements.txt
### Hardware

* A CUDA-capable GPU with at least **16 GB VRAM** is required for FP32 training at batch size 16 and sequence length 128 (XLM-RoBERTa-base requires approximately 14.6 GB peak GPU memory).
* Total runtime for all 32 runs is approximately 20--22 hours on an NVIDIA V100 GPU.

### Software

* Python 3.9 or newer
* Dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Required Input Files

Place the following files in the working directory (or pass their paths via command-line arguments):

* `train_dataset.csv`: Source training split containing `Tweets` and `label` (`0` or `1`) columns.
* `test_dataset.csv`: Source test split containing `Tweets` and `label` (`0` or `1`) columns.
* `stopwords.txt`: UTF-8 text file with one Urdu stop word per line.

The UCSA-20 dataset is publicly available from IEEE DataPort:
`https://ieee-dataport.org/documents/urdu-corpus-sentiment-analysis`

## Usage

Run the full 32-run benchmark:

```bash
python revised_experiment.py \
  --train-csv train_dataset.csv \
  --test-csv test_dataset.csv \
  --stopwords stopwords.txt \
  --out-dir outputs
```

If execution is interrupted, resume from completed runs in the same output directory:

```bash
python revised_experiment.py \
  --train-csv train_dataset.csv \
  --test-csv test_dataset.csv \
  --stopwords stopwords.txt \
  --out-dir outputs \
  --resume
```

## Outputs

All results are written to the directory specified by `--out-dir`:

* `dataset_audit.json`: Source row counts, removed duplicates, excluded train--test overlaps, and class distributions per split.
* `predictions/<model>_<condition>_seed<seed>.csv`: 32 files containing review-level ground truth, predicted labels, positive-class probabilities, error types (`FP`, `FN`, `correct`), negation marker flags, and word counts.
* `run_metrics.csv`: Test loss, confusion-matrix counts (`TN`, `FP`, `FN`, `TP`), runtime, and macro/per-class metrics for each of the 32 runs.
* `metric_summary.csv`: Four-seed mean, sample SD, and 95% $t$ confidence intervals per model and preprocessing condition.
* `paired_seed_tests.csv`: Seed-level paired $t$-tests and Holm-adjusted $p$-values for model comparisons and stop-word ablations.
* `paired_prediction_tests.csv`: Review-level paired bootstrap 95% confidence intervals and Holm-adjusted randomization $p$-values.
* `persistent_error_cases.csv`: Test reviews misclassified in the same direction across all four seeds for a given model and condition.

## Results Summary

On the fixed held-out test set ($n = 1{,}932$) with stop words retained across four seeds (`42`, `169`, `356`, `789`):

* **XLM-RoBERTa-base** achieved the highest performance at $83.792 \pm 0.300\%$ accuracy and $83.412 \pm 0.300\%$ macro-F1 (95% $t$ CI: $[82.935, 83.889]\%$).
* Mean macro-F1 for the remaining models was $81.504 \pm 0.600\%$ for **mBERT-cased**, $80.832 \pm 0.200\%$ for **mBERT-uncased**, and $80.592 \pm 0.200\%$ for **DistilBERT**.
* In paired review-level comparisons, XLM-RoBERTa-base exceeded the other three models by $1.908$--$2.820$ percentage points (Holm-adjusted randomization $p = 0.003$ for all three contrasts).
* **Stop-word ablation:** Retaining stop words increased mean macro-F1 by $0.765$--$1.218$ percentage points across all four models; after Holm adjustment, only mBERT-uncased met the $0.05$ significance threshold ($p = 0.042$).
