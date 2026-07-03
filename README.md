# FL Student Behavior TinyML

This repository provides a complete implementation of the paper **A Real-Time Federated TinyML Framework for Student Behavior Detection in Smart Classrooms**. It contains the full Python and shell code needed to prepare SCB-Dataset5, create IID and non-IID federated clients, train the centralized baseline, train the proximal federated model, run pruning and post-pruning fine-tuning, export deployment artifacts, benchmark host-side TFLite inference, evaluate checkpoints, run non-IID and scalability experiments, and compute communication cost.

## Dataset format

The code expects object-detection labels in YOLO text format. Each image must have one matching label file. Each label line must contain five values:

```text
class_id x_center y_center width height
```

Coordinates must be normalized between 0 and 1. Class IDs must be from 0 to 19.

Default dataset layout:

```text
datasets/SCB-Dataset5/
  images/
    image_000001.jpg
    image_000002.jpg
  labels/
    image_000001.txt
    image_000002.txt
```

The same structure can be used for UK_Dataset and SiTBehavior by changing the paths in `configs/paper_config.yaml`.

## SCB-Dataset5 classes

The default configuration uses the 20 classes described in the manuscript:

```text
hand_raising, reading, writing, bowing_head, turning_head, talking, guiding,
board_writing, standing, answering, stage_interaction, discussing, clapping,
yawning, screen, blackboard, teacher, leaning_on_desk, using_phone,
using_computer
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For TensorFlow Lite INT8 conversion, install the optional deployment converters:

```bash
pip install tensorflow onnx2tf
```

## Complete pipeline

Set the dataset paths in `configs/paper_config.yaml`, then run:

```bash
bash scripts/run_all.sh configs/paper_config.yaml
```

This executes the full workflow: split preparation, centralized training, federated training, pruning, fine-tuning, ONNX export, INT8 TFLite conversion, and TensorFlow Lite Micro C-array export.

## Main commands

Prepare train/validation/test splits and client manifests:

```bash
python scripts/00_prepare_splits.py --config configs/paper_config.yaml --alpha 0.5
```

Train the centralized FP32 baseline:

```bash
python scripts/01_train_centralized.py --config configs/paper_config.yaml
```

Train the five-client proximal federated model in IID mode:

```bash
python scripts/02_train_federated.py --config configs/paper_config.yaml --partition iid
```

Train the five-client proximal federated model in non-IID mode:

```bash
python scripts/02_train_federated.py --config configs/paper_config.yaml --partition noniid --alpha 0.5
```

Run 40% structured channel pruning and post-pruning fine-tuning:

```bash
python scripts/03_prune_finetune.py --config configs/paper_config.yaml
```

Export the compressed model to ONNX:

```bash
python scripts/04_export_onnx.py --config configs/paper_config.yaml
```

Convert ONNX to INT8 TensorFlow Lite:

```bash
python scripts/05_convert_int8_tflite.py --config configs/paper_config.yaml
```

Export the TFLite model to TensorFlow Lite Micro C-array source files:

```bash
python scripts/06_export_c_array.py --config configs/paper_config.yaml
```

Run host-side TFLite benchmarking:

```bash
python scripts/07_benchmark_tflite.py --config configs/paper_config.yaml
```

Evaluate a checkpoint:

```bash
python scripts/08_evaluate_checkpoint.py --config configs/paper_config.yaml --checkpoint outputs/checkpoints/federated_best.pt
```

Estimate communication cost for different client counts:

```bash
python scripts/09_communication_cost.py --config configs/paper_config.yaml
```

Evaluate on another YOLO-format dataset:

```bash
python scripts/10_cross_dataset_eval.py --config configs/paper_config.yaml --dataset-root datasets/UK_Dataset
```

Run a quick code sanity check:

```bash
python scripts/11_quick_sanity_check.py --config configs/paper_config.yaml
```

Run the ablation, non-IID sweep, and client scalability shell scripts:

```bash
bash scripts/run_ablation.sh configs/paper_config.yaml
bash scripts/run_federated_noniid_sweep.sh configs/paper_config.yaml
bash scripts/run_scalability.sh configs/paper_config.yaml
```

## Manuscript configuration implemented

- Input resolution: 320 × 320.
- Output tensor: 40 × 40 × 75.
- Number of classes: 20.
- Candidate boxes per grid cell: 3.
- Dataset split: 70% training, 15% validation, 15% testing.
- Federated clients: K = 5.
- Federated rounds: 50.
- Local epochs per round: 5.
- Local batch size: 8.
- Optimizer: AdamW.
- Federated learning rate: 5e-4.
- Weight decay: 1e-4.
- Proximal coefficient: μ = 0.01.
- Structured pruning ratio: 40%.
- Post-pruning fine-tuning: 20 epochs.
- Fine-tuning learning rate: 5e-5.
- INT8 calibration images: 512.
- Deployment target: 32-bit ARM Cortex-M-class microcontroller, 120 MHz, 256 KB SRAM, 1 MB Flash.

## Repository files

```text
configs/paper_config.yaml              Main experimental configuration
src/config.py                          Configuration and output paths
src/data/dataset.py                    YOLO dataset reader and preprocessing
src/data/splits.py                     Dataset scanning and 70/15/15 splitting
src/data/partition.py                  IID and Dirichlet non-IID partitioning
src/models/tiny_detector.py            Lightweight detection architecture
src/models/decode.py                   Prediction decoding and confidence filtering
src/losses/detection_loss.py           Detection loss
src/engine/train.py                    Training and evaluation loops
src/engine/federated.py                Proximal FL local training and FedAvg
src/compression/bn_folding.py          Batch-normalization folding
src/compression/pruning.py             Structured channel pruning utilities
src/deployment/export.py               ONNX, TFLite, and C-array export functions
src/deployment/benchmark.py            Host-side TFLite benchmark
src/experiments/communication.py       Communication cost estimation
src/utils/*.py                         Box, metric, checkpoint, and seed utilities
scripts/*.py                           Python entry points for each experiment
scripts/*.sh                           Shell entry points for full experiments
firmware/tflite_micro_main.cpp         TensorFlow Lite Micro inference application
```

## Outputs

The scripts write experiment artifacts under `outputs/` according to `configs/paper_config.yaml`. The training scripts create the checkpoint files after the dataset is available and the experiments are executed.
