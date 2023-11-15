### NUMBER DETECTION FILE STRUCTURE TREE

- number_detection/
  - archive/              # Old files
  - datasets/
    - train/              # Training dataset (preprocessed)
    - test/               # Test dataset (unprocessed raw images)
    - val/                # Validation dataset (preprocessed)
  - src/
    - preprocessing.py    # Script for preprocessing 'raw' images on-the-fly
    - training.py         # Script for training the model, starting with MNIST
    - inference.py        # Script for applying the model to new images
    - testing.py          # Script for evaluating the trained model
  - models/
    - cnn_model.py        # Model architecture
    - saved_models/       # Where trained PyTorch models are saved