# Exercise Form Classification with LSTM

Final project for CCS 248: Artificial Neural Networks. Using wearable sensor data to classify biceps curl exercise form quality.

## Classification

Classifies exercise form into 5 categories using LSTM neural network:
- **Class A**: Correct form
- **Class B**: Elbows too far forward
- **Class C**: Lifting only halfway
- **Class D**: Lowering only halfway
- **Class E**: Hips too far forward

## Dataset

- **Source**: Weight Lifting Exercises Dataset (UCI)
- **Size**: 4,024 exercise repetitions
- **Features**: 52 sensor readings (accelerometer, gyroscope, magnetometer)
- **Sensors**: Belt, arm, forearm, and dumbbell

## Model Architecture

**LSTM-based classifier:**
- 2-layer LSTM (128 hidden units)
- Fully connected layers (128 → 64 → 5)
- Dropout (0.5) to prevent overfitting

## Technologies Used

- **PyTorch**: Model building and training
- **Pandas/NumPy**: Data processing
- **scikit-learn**: Data splitting and evaluation
- **Matplotlib/Seaborn**: Visualization

## Training Setup

**Data Split:**
- Training: 70%
- Validation: 15%
- Test: 15%

**Best Hyperparameters:**
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 50 (with early stopping)
