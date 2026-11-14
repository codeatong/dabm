# DABM-Net

DABM-Net detects telecom fraud by combining an hour–date behavioral tensor with
a vector of user-level context features. The default configuration uses eight
temporal channels covering call and SMS counts by type, call duration, and
distinct contact counts, plus ten global features including regional diversity,
data usage, and average revenue per user (ARPU) statistics.

DSFF uses each user's global context to reweight the temporal features. CBMC
combines activity levels, temporal changes, and local deviations into an
anomaly-aware hour–date matrix.
DTTI-Net adaptively combines patterns across consecutive hours and across days
at the same hour, along with the discrepancy between the two representations,
to estimate each user's fraud probability.

## Running the code

Run the commands from the repository root.

### 1. Install dependencies
<!--
Use Python 3.8.19.
python3.8 -m venv .venv
source .venv/bin/activate
-->
```bash
python -m pip install -r requirements.txt
```

### 2. Prepare the data
Use the dataset files in the `dataset/` directory and then run the code for data processing.
If the dataset can notdownload, please contact the corresponding author for authorized access to the complete data. 

<!--Place `voc_merge_all.csv` (calls), `train_sms.csv` (SMS), and
`all_feat_new.csv` (user features and labels) in `dataset/`.
-->
<!--
python -m dabm.data_process \
  --temporal-output dataset/calendar_temporal_tensor.npy \
  --metadata-output dataset/calendar_metadata.npz
-->

```bash
python -m dabm.data_process
```

If you already have both generated files in `dataset/`, go directly to training.

### 3. Train and evaluate
Execute the following command to train the model, and you will obtain the final result. 
The parameter settings have been provided in the article. 
You can also adjust the parameters and re-train the model.

<!--Run the 80/20 telecom experiment with seed 42. The trainer evaluates the
final checkpoint on the held-out test users automatically. -->
<!--python -m dabm.train_dabm_net \
  --temporal-path dataset/calendar_temporal_tensor.npy \
  --metadata-path dataset/calendar_metadata.npz \
  --protocol train80 --seed 42 --split-seed 42 \
  --target-training-instances 12000 --test-time-views 3 \
  --device cuda --output-dir runs/seed_42 
  -->
```bash
python train_dabm_net.py \
  --protocol train80 \
  --epochs 500 \
  --batch-size 16 
```

### 4. View the results
The result after running will be saved in the specified path (if specified), 
and a corresponding csv file will be generated.

<!--Open `runs/seed_42/comparison_summary.csv` for the test metrics. The
`runs/seed_42/train_80_test_20_aug_12000_tta_3/` directory contains
`final_model.pt`, `split_indices.npz`, and `test_predictions.csv`.
-->
