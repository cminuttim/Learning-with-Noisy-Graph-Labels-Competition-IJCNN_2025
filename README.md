# IJCNN 2025 Competition: Learning with Noisy Graph Labels

This repository contains the [winning solution](https://sites.google.com/view/learning-with-noisy-graph-labe/winners) of the challenge [IJCNN 2025 Competition: Learning with Noisy Graph Labels](https://sites.google.com/view/learning-with-noisy-graph-labe?usp=sharing). The approach leverages a **Variational Graph Autoencoder (VGAE)** to filter noisy data, a **ensemble of models** strategy to handle different types of noise, and a **weighted voting mechanism** to improve prediction accuracy.

---

## Overview of the Method

The method consists of four key components to handle noisy labels effectively:

1. **Variational Graph Autoencoder (VGAE):**
   - The VGAE is used to filter noisy data by retaining only real patterns in the bottleneck. This is analogous to PCA but operates nonlinearly, making it suitable for complex graph-structured data.

2. **Dropout Regularization:**
   - A 5% dropout is applied to prevent the model from over-relying on potentially noisy features, ensuring robustness.

3. **Simulated Weak Pretraining and Data Augmentation:**
   - A general model is pretrained on all datasets (A, B, C, D) to emulate weak pretraining and large-scale data augmentation. This helps the model generalize better across different types of noise.

4. **Weighted Ensemble of Models:**
   - Multiple models are trained on different subsets of the data, each potentially capturing different noise patterns. The final prediction is a weighted average of the predictions from these models, with weights determined by their F1 scores. This ensemble approach improves robustness to noise and enhances prediction accuracy.

---

## Procedure

1. **Data Preparation:**
   - The datasets (A, B, C, D) are loaded and preprocessed into graph structures.
   - Each graph is represented with node features, edge indices, and edge attributes.

2. **Initial Pretraining on All Datasets:**
   - The model is first pretrained on all datasets (A, B, C, D) to learn general patterns and noise characteristics. This pretraining acts as a form of weak supervision and data augmentation, allowing the model to generalize better when fine-tuned on individual datasets.
   - Example command for pretraining:
     ```bash
     python main.py --train_path "../A/train.json.gz ../B/train.json.gz ../C/train.json.gz ../D/train.json.gz" --num_cycles 5
     ```
   - This generates a pretrained model file (e.g., `model_paths_ABCD.txt`) that can be used for fine-tuning on specific datasets.

3. **Fine-Tuning on Individual Datasets:**
   - After pretraining, the model is fine-tuned on individual datasets (e.g., dataset A) using the pretrained model as a starting point. This allows the model to adapt to the specific noise patterns of the target dataset while retaining the general knowledge learned during pretraining.
   - Example command for fine-tuning on dataset A:
     ```bash
     python main.py --train_path ../A/train.json.gz --test_path ../A/test.json.gz --num_cycles 5 --pretrain_paths model_paths_ABCD.txt
     ```

4. **Prediction:**
   - The ensemble of models is used to predict on the test set.
   - Predictions from each model are combined using weighted voting, where the weights are the F1 scores of the models.

5. **Resuming Training:**
   - Training can be resumed by loading pretrained models and continuing training with adjusted hyperparameters (e.g., learning rate, batch size).

---

## Usage

### Pretraining on All Datasets
To pretrain the model on all datasets (A, B, C, D) for 5 cycles:
```bash
python main.py --train_path "../A/train.json.gz ../B/train.json.gz ../C/train.json.gz ../D/train.json.gz" --num_cycles 5
```

### Fine-Tuning on a Specific Dataset
To fine-tune the model on dataset A using the pretrained model (`model_paths_ABCD.txt`):
```bash
python main.py --train_path ../A/train.json.gz --test_path ../A/test.json.gz --num_cycles 5 --pretrain_paths model_paths_ABCD.txt
```

### Prediction
To make predictions using the trained models:
```bash
python main.py --test_path ../A/test.json.gz --pretrain_paths model_paths_A.txt
```

### Resuming Training
To resume training with pretrained models:
```bash
python main.py --train_path ../A/train.json.gz --test_path ../A/test.json.gz --num_cycles 5 --pretrain_paths model_paths_A.txt
```

---

## Key Components

### 1. Variational Graph Autoencoder (VGAE)
- **Encoder:** The encoder maps the input graph to a latent space, capturing the essential patterns while filtering out noise.
- **Decoder:** The decoder reconstructs the graph from the latent space, ensuring that the learned representations are meaningful.
- **Loss Function:** The loss function combines reconstruction loss, KL divergence, and classification loss to train the model effectively.

### 2. Ensemble of Models and Weighted Voting
- The predictions from the ensemble of models are combined using weighted voting, where the weights are the F1 scores of the models. This ensures that models with better performance contribute more to the final prediction.

---

## Code Structure

- **`main.py`:** The main script for training, evaluating, and predicting with the model.
- **`EdgeVGAE`:** The core model class implementing the VGAE with a classification head.
- **`ModelTrainer`:** A utility class for training multiple cycles and managing the ensemble of models.
- **`Config`:** A configuration class for managing hyperparameters and settings.

---

## Results

The method achieves robust performance across different datasets with varying types of noise. The use of VGAE for noise filtering, combined with the ensemble of models and weighted voting, ensures that the model generalizes well and produces accurate predictions.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact Carlos Minutti at cminutti@data-fusionlab.com
