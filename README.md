# FL Student Behavior TinyML

This repository provides a complete implementation of the paper **A Real-Time Federated TinyML Framework for Student Behavior Detection in Smart Classrooms**. It contains the full Python and shell code needed to prepare SCB-Dataset5, create IID and non-IID federated clients, train the centralized baseline, train the proximal federated model, run pruning and post-pruning fine-tuning, export deployment artifacts, benchmark host-side TFLite inference, evaluate checkpoints, run non-IID and scalability experiments, and compute communication cost.


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
