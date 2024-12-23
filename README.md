# ChromBPNet PyTorch Lightning Implementation

A PyTorch Lightning reimplementation of ChromBPNet developed during fall BMI rotation with the Corces Lab. ChromBPNet is a deep learning model for predicting chromatin accessibility from DNA sequence.

## Project Structure

### `models/`
- `base_modules.py`: Contains the fundamental building blocks of the neural network architecture
- `_module.py`: Arranges the base modules into the complete model architecture
- `_model.py`: Contains the PyTorch Lightning trainer class implementation
- `_data.py`: Implements data loading and preprocessing functionality

### `testing/`
- `notebooks/`: Jupyter notebooks used for model development, testing, and comparisons with the original ChromBPNet implementation
- `microglia_train.sh` and `microglia_train_new.sh`: Training scripts for microglia-specific experiments

### `utils/`
- `adjust_bed.py`: Utilities for handling BED file format adjustments
- `attention_utils.py`: Helper functions for attention mechanisms
- `data_utils.py`: General data processing utilities
- `losses.py`: Loss function implementations (Note: NLL implementation currently in progress)
- `one_hot.py`: Functions for one-hot encoding of DNA sequences
- `shape_utils.py`: Utilities for handling tensor shapes and transformations


## Development TODOs:

- [ ] Complete NLL loss implementation
- [ ] Personalized Genome integration (paired WGS, scATAC-Seq)
