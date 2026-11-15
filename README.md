# Nepal ScamShield

An AI-assisted SMS threat-awareness application for reviewing suspicious messages. The current classifier is trained on UCI English SMS spam and a separate Bengali-focused multilingual smishing dataset. Nepal-focused synthetic examples are evaluated separately; benchmark metrics apply to the documented corpus splits.

## Project overview

ScamShield combines a saved message classifier, an adjustable spam threshold, and independent message-pattern indicators in one SMS-awareness application. The unified corpus preserves each source's original category while mapping examples to `ham` or `spam` for evaluation. Nepal-focused synthetic examples are reported separately and were not used to fit or select the model.

## Sources, licences, and limitations

### UCI SMS Spam Collection

Almeida and Hidalgo (2011), [UCI Machine Learning Repository: SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84), 5,574 English SMS rows with `ham` and `spam` labels. UCI identifies the licence as CC BY 4.0 and credits several underlying corpora and sources. The source contains repeated texts; duplicates were removed before splitting. Attribution is included in [data/source/README.md](data/source/README.md).

### Bengali SMS Smishing Dataset

[Hugging Face dataset card](https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset), 7,005 rows in its published train/validation/test partitions. The card declares MIT and lists Bengali, English, Banglish, and CodeMix varieties, with original categories `smish`, `promo`, and `normal`. `smish` maps to `spam`; `promo` and `normal` map to `ham`, because the card describes promotion as legitimate marketing. Original labels and language variety are retained in the unified CSVs. The related [research description](https://researchportal.murdoch.edu.au/esploro/outputs/graduate/A-Multilingual-Fine-Tuned-LLM-Framework-for/991005909771807891) reports LLM translation and synthetic generation in dataset construction. The repository-level MIT declaration is recorded, but the underlying source of every row is not independently auditable. Results measure this published corpus and its generation and translation patterns. Bengali/Bangla and Banglish are not Nepali.

### Financial scams detection dataset

Ahmed and Islam (2025), [Mendeley Data, version 2](https://data.mendeley.com/datasets/znsk27yk3h/2), DOI [10.17632/znsk27yk3h.2](https://doi.org/10.17632/znsk27yk3h.2), declares CC BY 4.0. Its public description says messages are manually labeled `scam`/`ham`, English and Bangla, and sourced across SMS, chat, and email. The file could not be reliably retrieved here (public API returned 401; record download returned 403), so it is not included and no results are claimed for it. Its channel mix also needs review before treating it as SMS-only.

### Synthetic and exploratory data

`data/nepal_synthetic_test.csv` contains 16 manually written Nepal-focused examples and remains a separate exploratory evaluation only. It is excluded from training and selection. `data/multilingual_synthetic_exploratory.csv` is retained as an exploratory file but excluded from training and evaluation because its provenance is unverified and its contents do not establish Nepal relevance. Neither synthetic file represents scam prevalence or reported incidents.

## Dataset preparation and quality checks

Run `python prepare_dataset.py` after installing dependencies. The script reads the downloaded source artifacts in `data/source/`, removes empty messages and conflicting-label duplicate groups, then de-duplicates on Unicode NFKC + casefolded + whitespace-collapsed text (punctuation retained). It preserves source dataset, original category, language variety, source partition, and source row ID.

The source artifacts contain 5,574 UCI rows (`ham` 4,827; `spam` 747) and 7,005 Bengali-dataset rows (`normal` 2,488; `promo` 1,708; `smish` 2,809). The recorded preparation run found 0 empty messages, 0 conflicting-label duplicate groups, and removed 424 exact duplicate rows from 12,579 source rows (415 UCI rows and 9 Bengali-dataset rows). It found zero exact normalized-text overlaps between each pair of effective splits. This exact-text check does not detect near-duplicate templates or campaign-level overlap, so it cannot rule out every form of leakage. Detailed counts and source-file SHA-256 hashes are in `data/dataset_audit.json`.

The Bengali dataset's published test partition remains held out. UCI has no official split here, so its de-duplicated rows were split with seed 42 into 70% train, 10% validation, and 20% test. Final unified counts:

| Split | Rows | Ham | Spam |
| --- | ---: | ---: | ---: |
| Train | 8,505 | 6,092 | 2,413 |
| Validation | 1,217 | 871 | 346 |
| Final test | 2,433 | 1,744 | 689 |

The final test consists of the Bengali publisher's held-out test partition and the seeded UCI holdout. Duplicate text is assigned to at most one effective split. Nepal synthetic examples are not used for any fit or tuning.

## Model and evaluation protocol

`train_models.py` fits TF-IDF word unigrams and bigrams (`min_df=2`, sublinear TF) with Logistic Regression, Multinomial Naive Bayes, and sigmoid-calibrated Linear SVM. The fixed probability threshold is 0.50. Model selection uses validation spam recall, with ties broken by spam precision then F1. All three fixed candidates are then reported on the final test for comparison; those test results do not select or tune the model. The chosen model is refit on train plus validation and saved to `model/nepal_scamshield_unified.pkl`.

Metrics are accuracy and spam-class precision, recall, and F1. Confusion matrix order is rows actual `[ham, spam]`, columns predicted `[ham, spam]`. False positives are benign messages flagged as spam; false negatives are spam messages missed.

### Validation comparison (selection split, n=1,217)

| Model | Accuracy | Spam precision | Spam recall | Spam F1 | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 97.12% | 97.55% | 92.20% | 94.80% | `[[863, 8], [27, 319]]` |
| Multinomial Naive Bayes | 96.63% | 96.92% | 91.04% | 93.89% | `[[861, 10], [31, 315]]` |
| Linear SVM (sigmoid-calibrated) | 98.36% | 95.79% | 98.55% | 97.15% | `[[856, 15], [5, 341]]` |

Linear SVM was selected on validation because it had the highest spam recall at the fixed threshold.

### Final test comparison (n=2,433)

| Model | Accuracy | Spam precision | Spam recall | Spam F1 | FP | FN | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 97.12% | 97.54% | 92.16% | 94.78% | 16 | 54 | `[[1728, 16], [54, 635]]` |
| Multinomial Naive Bayes | 96.55% | 96.04% | 91.58% | 93.76% | 26 | 58 | `[[1718, 26], [58, 631]]` |
| Linear SVM (selected) | 98.68% | 97.27% | 98.11% | 97.69% | 19 | 13 | `[[1725, 19], [13, 676]]` |

The selected model's source/language-specific metrics and confusion matrices are in `data/model_evaluation.json` and shown in the app.

| Final test source / language | n | Accuracy | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI — English | 1,032 | 99.03% | 99.17% | 92.97% | 95.97% |
| Bengali dataset — Bengali | 358 | 98.04% | 95.89% | 99.29% | 97.56% |
| Bengali dataset — Banglish | 360 | 98.33% | 96.55% | 99.29% | 97.90% |
| Bengali dataset — CodeMix | 336 | 97.92% | 95.80% | 99.28% | 97.51% |
| Bengali dataset — English | 347 | 99.42% | 99.29% | 99.29% | 99.29% |

These are separate corpus/language slices and do not establish Nepali performance.

These scores describe the named, deduplicated corpora and their partitions. UCI's `spam` label covers general SMS spam, while Bengali `smish` labels smishing and `promo` labels legitimate marketing. Original categories remain in the data, and the combined benchmark is a mixed-source binary task. The benchmark measures these corpus splits; it does not provide a Nepal-specific performance estimate.

### Nepal synthetic exploratory evaluation

`python evaluate_nepal.py` scores 16 hand-written Nepal-focused examples with the saved unified model. Current result: accuracy 100%, spam precision 100%, spam recall 100%, F1 100%; confusion matrix `[[8, 0], [0, 8]]`. This result summarizes those authored examples only. The set is separate from the benchmark and is never used to train, choose, or tune the model.

## Run locally

Use Python 3.12 or 3.13 with the pinned dependencies. From the repository root:

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python evaluate_nepal.py
streamlit run app.py
```

On macOS/Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate` in place of the first two commands. Evaluation loads the checked-in model and reports the separate Nepal synthetic set; starting the app requires no training step. `prepare_dataset.py` and `train_models.py` are reproducibility tools. Running `train_models.py` refits the three candidates and overwrites the saved unified model and benchmark report, so it is not part of the normal launch flow.

The app and evaluation were exercised in the available Python 3.14.5 environment with Streamlit 1.67.0, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1, PyArrow 26.0.1, and joblib 1.6.0. The exact NumPy 2.4.6 and pandas 2.2.3 pins have not been installed and validated together in a clean environment.


## Technology and responsible use

Python, Streamlit, scikit-learn, TF-IDF, Logistic Regression, Multinomial Naive Bayes, Linear SVM, pandas, NumPy, PyArrow, and joblib. The app adds rule-based URL/keyword indicators, kept distinct from the machine-learning score. Feature contributions show learned TF-IDF associations and provide context for the model score.

Use the score to prioritize review, and verify sensitive requests through official channels before taking action. Do not use the score as the sole basis for accusing a sender or making a financial decision. Avoid entering private messages or sensitive data in public demos. This application provides message-awareness guidance and is not financial advice.
