# Choice Problem (Fracktional Knapsack) — AINN Blueprint

This folder contains an implementation of **Choice Problem** using **Fractional Knapsack** as the algorithmic blueprint.

## Prerequisites

### CLRS
This project requires related packages of **CLRS**. Please install these packages following their official [instructions](https://github.com/google-deepmind/clrs).

### Other dependencies
Most other dependencies can be easily installed via `pip`. A `requirements.txt` is provided in the **parent directory** as a reference—feel free to adjust package versions to match your hardware/software environment.

---

## Train AINN

Here we provide a Bottom-up approach for AINN training.
Run training with the default AINN configuration:

```bash
python3 train_ainn_stage1.py --config ainn
python3 train_ainn_stage2.py --config ainn
python3 train_ainn_stage3.py --config ainn
```

Configuration files live in `./config` as YAML. The command above uses the default config, but you can edit the YAML to tune hyperparameters for your setup.
You can tune the `num_samples` parameter to control the size of the training dataset (E.g., 1000, 2000, etc.)
All the models are saved in `./ainn_model` and `./tmp/CLRS30`.
Right now, this training uses a PSO method that runs mostly on the CPU. We will release a CUDA/GPU version soon.

---

## Train Monitor (Encoders for Debuggability)

To train the monitoring encoders used for debugging, we first train a **lower-accuracy AINN** (as described in the paper) to make debugging behavior more observable.

### 1) Train a reduced AINN

```bash
python3 train_ainn_stage1.py --config ainn_less_iter
python3 train_ainn_stage2.py --config ainn_less_iter
python3 train_ainn_stage3.py --config ainn_less_iter
```

This uses `ainn_less_iter.yaml` from the `config/` folder by default.

### 2) Prepare the monitoring dataset and train encoders

```bash
cd monitor

# Generate intermediate representations with the same config
python3 train_gen_monitor_val_stage1.py --config ainn_less_iter
python3 train_gen_monitor_val_stage2.py --config ainn_less_iter
python3 train_gen_monitor_val_stage3.py --config ainn_less_iter

# Train outlier encoders
python3 outlier_stage1_fit_curve.py
python3 outlier_stage2.py
python3 outlier_stage3.py
```

All the encoder models are saved in `./monitor/model`.

---

## Train Baseline

Baseline code is under `./baselines`. To train:

```bash
cd baseline

python3 train_baseline.py --num_samples <number of samples>
```

The default number of training samples is `1000`. (if you run without using flag)
