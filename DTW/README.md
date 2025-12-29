# Keyword Spotting (DTW + KNN)

This folder contains an implementation of **Keyword Spotting (KWS)** using **DTW + KNN** as the algorithmic blueprint.

## Prerequisites

### torchsort
This project requires **torchsort**. Please install it following the official [instructions](https://github.com/teddykoker/torchsort).

### Other dependencies
Most other dependencies can be easily installed via `pip`. A `requirements.txt` is provided in the **parent directory** as a reference—feel free to adjust package versions to match your hardware/software environment.

---

## Train AINN

Here we provide a Top-down approach for AINN training.
Run training with the default AINN configuration:

```bash
python3 train_ainn_join.py --config ainn
```

Configuration files live in `./config` as YAML. The command above uses the default config, but you can edit the YAML to tune hyperparameters for your setup.
Note that, unlike the other two applications that use `num_samples` to control dataset size, this multi-class setting uses a more compact parameter--`portion`. You can adjust this `portion` to choose how much data to use for training; the script will automatically load the corresponding dataset from the `./dataset` folder. (E.g., 0.1 means using only 10% of the data for training.)
All the models are saved in `./ainn_model`.

---

## Train Monitor (Encoders for Debuggability)

To train the monitoring encoders used for debugging, we first train a **lower-accuracy AINN** (as described in the paper) to make debugging behavior more observable.

### 1) Train a reduced AINN

```bash
python3 train_ainn_join.py --config ainn_less_iter
```

This uses `ainn_less_iter.yaml` from the `config/` folder by default.

### 2) Prepare the monitoring dataset and train encoders

```bash
cd monitor

# Generate intermediate representations with the same config
python3 train_gen_monitor_val.py --config ainn_less_iter

# Postprocess representations with labels
python3 construct_pos_neg.py --mode train
python3 construct_pos_neg.py --mode validation
python3 construct_pos_neg.py --mode test

# Train outlier encoders
python3 outlier_encoders.py
```

All the encoder models are saved in `./monitor/model`.

---

## Train Baseline

Baseline code is under `./baselines`. To train:

```bash
python3 train_mobi_baseline.py --portion <portion>
```

The default `portion` is `0.1`. (if you run without using flag)
