# Fall Detection (HMM) — AINN Blueprint

This folder contains an implementation of **Fall Detection** using **HMM** as the algorithmic blueprint.

## Prerequisites

### Dependencies
Relevant dependencies can be easily installed via `pip`. A `requirements.txt` is provided in the **parent directory** as a reference—feel free to adjust package versions to match your hardware/software environment.

---

## Train AINN

Here we provide a bottom-up approach for AINN trainnig.
Run training with the default AINN configuration:

```bash
python3 train_ainn_hmm_stage1.py --config ainn
python3 train_ainn_hmm_stage2.py --config ainn
```

Configuration files live in `./config` as YAML. The command above uses the default config, but you can edit the YAML to tune hyperparameters for your setup.
You can tune the `num_samples` parameter to control the size of the training dataset. (E.g., 50, 100, 200, etc.)
All the models are saved in `./ainn_model`.
Right now, this training uses a PSO method that runs mostly on the CPU. We will release a CUDA/GPU version soon.

---

## Train Monitor (Encoders for Debuggability)

To train the monitoring encoders used for debugging, we first train a **lower-accuracy AINN** (as described in the paper) to make debugging behavior more observable.

### 1) Train a reduced AINN

```bash
python3 train_ainn_hmm_stage1.py --config ainn_less_iter
python3 train_ainn_hmm_stage2.py --config ainn_less_iter
```

This uses `ainn_less_iter.yaml` from the `config/` folder by default.

### 2) Prepare the monitoring dataset and train encoders

```bash
cd monitor

# Generate intermediate representations with the same config
python3 train_gen_monitor_val.py --config ainn_less_iter

# Train outlier encoders
python3 outlier_encoders.py
```

All the encoder models are saved in `./monitor/model`.

---

## Train Baseline

Baseline code is under `./baselines`. To train:

```bash
cd baselines

python3 train_BRNN.py --num_samples <number of samples>
python3 train_MLP.py --num_samples <number of samples>
```

The default number of training samples is `50`. (if you run without using flag)
