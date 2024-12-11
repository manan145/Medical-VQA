# Medical-VQA
This project fine-tunes advanced multimodal models, ViLT and BLIP-2, for medical Visual Question Answering (VQA) using the SLAKE dataset. It evaluates performance with F1 and Exact Match metrics, showcasing how LoRA fine-tuning and BLIP-2’s modular design outperforms ViLT for domain-specific tasks.

## **Key Components**

### 1. **Dataset Preparation**
- **Dataset**: The **SLAKE dataset** was used for medical Visual Question Answering (VQA).
- **Processing**:
  - Images were preprocessed using **Pillow** and encoded for model input with `ViltProcessor` (ViLT) and `AutoProcessor` (BLIP-2).
  - Questions were tokenized into natural language prompts for BLIP-2 model.
  - Answers were mapped to unique labels using a `label2id` dictionary for classification compatibility of ViLT model.
- **Utilities**:
  - `data.py`: Handles loading, preprocessing, and splitting of the SLAKE dataset into training, validation, and test sets.
  - `utils.py`: Includes helper functions for image resizing, normalization, and tokenized question-answer pairing.

### 2. **Model Fine-Tuning**
- **ViLT**:
  - Baseline model with lightweight design.
  - Divides images into patches, flattens them, and passes through a transformer encoder.
  - Fine-tuned for medical VQA with a classification head.
- **BLIP-2**:
  - Combines a frozen vision encoder, Querying Transformer (Q-Former), and a large language model.
  - Fine-tuned using **LoRA (Low-Rank Adaptation)** for efficient parameter updates.

### 3. **Training Configuration**
- Utilized **Google Colab Pro** with **A100 GPU** for accelerated training.
- **LoRA Fine-Tuning**:
  - Implemented **Low-Rank Adaptation (LoRA)** for BLIP-2 to enable parameter-efficient training by freezing most of the pre-trained parameters and updating only a small number of additional trainable weights in the Querying Transformer (Q-Former).
  - LoRA reduces memory usage and speeds up training while maintaining performance.
  - Configuration:
    - Rank (`r`): 8
    - Alpha: 32
    - Dropout: 0.1
- **Hyperparameters**:
  - Batch size: 64
  - Learning rate: 2e-3 (BLIP-2) | 2e-5 (ViLT)
  - Epochs: 10 (BLIP-2) | 3 (ViLT)
- Used **mixed-precision training (FP16)** to optimize resource usage.

### 4. **Evaluation**
- Metrics: **SQuAD v2 metrics** (Exact Match and F1 score).
  - **Exact Match (EM)**: Measures the percentage of predictions that exactly match the ground truth.
  - **F1 Score**: Balances precision and recall to evaluate partial matches.
- Results:
  - **ViLT**: F1 score of 0.7166.
  - **BLIP-2**:
    - F1 score: **0.7957**.
    - Exact Match (EM): **75.77%**.

### 5. **Building a Front-End Application**
- Developed a **VQA App** using **Streamlit** for an interactive and user-friendly interface.
- Features:
  - Upload medical images and input questions for real-time inference.
  - Visualize model predictions and responses dynamically.
  - Accessible and deployable for demonstrations or real-world use cases.
