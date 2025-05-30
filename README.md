MLP with Batch Normalization

This Jupyter Notebook (makemore_part3_bn.ipynb) implements a character-level language model using a Multi-Layer Perceptron (MLP) with batch normalization to generate names from a dataset. It builds on previous "makemore" tutorials, introducing batch normalization to stabilize and improve training. This README outlines the notebook's purpose, structure, and key components.

Overview

The notebook trains an MLP to predict the next character in a sequence, given a context of prior characters, using names from names.txt. Batch normalization is added to normalize hidden layer activations, improving training stability and speed. The model is trained, evaluated, and used to generate sample names.

Key Components

1. Dataset





Source: names.txt (32,033 names, e.g., "emma", "olivia").



Vocabulary: 27 characters (26 letters + '.' as a special token).



Context: Block size of 3 characters to predict the next one.



Splits:





Training: 80% (182,625 examples)



Validation: 10% (22,655 examples)



Test: 10% (22,866 examples)



Process: Maps characters to integers (stoi, itos), builds input (X) and target (Y) tensors.

2. Model Architecture





Embedding Layer:





Maps each character to a 10-dimensional vector (n_embd = 10).



C: Embedding matrix (27, 10).



Hidden Layer:





Linear layer: Input size n_embd * block_size (30), output size n_hidden = 200.



Weights W1 initialized with gain (5/3)/(input_size^0.5).



Batch Normalization:





Normalizes pre-activations: (hpreact - mean) / std + bnbias, scaled by bngain.



Running mean (bnmean_running) and std (bnstd_running) updated with momentum (0.999, 0.001).



Non-Linearity: Tanh activation applied to normalized pre-activations.



Output Layer:





Linear layer: Maps 200 hidden units to 27 classes (vocab size).



Weights W2 and bias b2 initialized with small values.



Parameters: Total 12,097, including C, W1, W2, b2, bngain, bnbias.

3. Training





Optimizer: Gradient descent with learning rate 0.1 (first 100,000 steps), then 0.01.



Batch Size: 32 examples.



Steps: 200,000 iterations.



Loss: Cross-entropy loss via F.cross_entropy.



Process:





Forward pass: Embedding → Concatenation → Linear → BatchNorm → Tanh → Linear → Loss.



Backward pass: Computes gradients, updates parameters.



Tracks loss for monitoring (printed every 10,000 steps).

Notebook Structure





Setup:





Imports torch, torch.nn.functional, matplotlib.



Loads names.txt, builds vocabulary, and creates datasets (Xtr, Ytr, etc.).



Initializes model parameters with a fixed seed (2147483647) for reproducibility.



Training Loop:





Constructs mini-batches, performs forward and backward passes.



Updates running batch norm stats without gradient tracking.



Logs loss for visualization.



Visualization:





Plots parameter update ratios (not fully implemented in code; ud undefined).



Intended to check gradient-to-update ratios (~1e-3).



Evaluation:





Computes train and validation losses using a split_loss function.



Results: Train loss ~2.40, Validation loss ~2.40.



Sampling:





Generates 20 names by sampling from the model’s softmax probabilities.



Examples: carpah., qarlileif., jmrix., etc.



Bonus Content:





Interactive widget to visualize batch normalization’s effect on input distribution.



Experiments with activation and gradient statistics for linear and batch norm layers.

Requirements





Python 3.x



Libraries:





torch (PyTorch for tensors and neural network operations)



matplotlib (for plotting loss and visualizations)



ipywidgets, scipy, numpy (for bonus interactive widget)



Dataset: names.txt (place in the same directory)

Usage





Clone the repository or download the notebook.



Ensure names.txt is in the working directory.



Install required libraries: pip install torch matplotlib ipywidgets scipy numpy.



Open makemore_part3_bn.ipynb in Jupyter Notebook or JupyterLab.



Run cells sequentially to:





Load and preprocess data.



Train the model.



Evaluate on train and validation sets.



Generate sample names.



Explore bonus visualizations.

Results





Train Loss: ~2.4003



Validation Loss: ~2.3982



Sample Outputs: Generated names like carpah., kalein., arian., etc., showing plausible but imperfect name-like structures.

Notes





Batch normalization stabilizes training by normalizing hidden layer pre-activations.



The bonus section explores activation and gradient statistics, highlighting batch norm’s effect on variance.



The plot cell references an undefined ud variable, suggesting a missing update-tracking step.



Model performance is reasonable but not optimized for high-quality name generation.
