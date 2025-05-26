**Makemore Part 1: Bigram Language Model**<br/>
**Overview**
This repository contains the implementation of a bigram language model as part of the "Makemore" series, inspired by Andrej Karpathy's neural network tutorials. The model is built using PyTorch and trained on a dataset of names to predict the next character in a sequence based on the current character, effectively generating new names. This project demonstrates the basics of neural network training, including one-hot encoding, matrix multiplication, softmax, and gradient descent optimization.
The code is provided in a Jupyter notebook (makemore_part1_bigrams.ipynb) and focuses on:

Loading and preprocessing a dataset of names.
Building a bigram model by counting character transitions.
Training a neural network to learn bigram probabilities.
Sampling from the trained model to generate new names.
<br/>
**Installation**
To run the code, you need Python 3.6+ and the following dependencies:
pip install torch matplotlib

**Prerequisites**

Python: Version 3.6 or higher.
Jupyter Notebook: To run the .ipynb file interactively.
Dataset: The names.txt file containing a list of names (one per line) should be in the project directory.

**Clone the repository:**
git clone https://github.com/your-username/makemore-part1-bigrams.git<br/>
cd makemore-part1-bigrams
**
Usage

Prepare the Dataset:**

Ensure names.txt is in the project root directory. This file contains a list of names used for training the model.
The notebook expects the dataset to be in the same directory as makemore_part1_bigrams.ipynb.


**Run the Notebook:**

Open the Jupyter notebook:jupyter notebook makemore_part1_bigrams.ipynb


Execute the cells sequentially to:
Load and explore the dataset.
Compute bigram frequencies.
Train the neural network model.
Generate sample names.




**Key Sections in the Notebook:**

Data Loading: Reads names.txt and splits it into a list of words.
Bigram Counting: Creates a dictionary of bigram frequencies with special "(S) and (E)" tokens for start and end.
Neural Network:
One-hot encodes input characters.
Uses a weight matrix (W) to predict logits.
Applies softmax to obtain probabilities.
Computes negative log likelihood loss.
<br/>

Training: Performs gradient descent with L2 regularization to optimize the weights.
Sampling: Generates new names by sampling from the trained model's probability distribution.

<br/>
Sample Output:After training, the model generates names like:
mor.
axx.
minaymoryles.
kondlaisah.
anchthizarie.

<br/>

**Dataset**
The dataset (names.txt) contains 32,033 unique names, one per line. The names range in length from 2 to 15 characters. The model treats each name as a sequence of characters, augmented with special tokens (. for start and end) to model bigram transitions.
Example names from the dataset:
emma
olivia
ava
isabella
sophia
<br/>
**Project Structure**
makemore-part1-bigrams/<br/>
│<br/>
├── makemore_part1_bigrams.ipynb  # Main Jupyter notebook with the bigram model <br/>
├── names.txt                     # Dataset of names (not included, add your own) <br/>
├── README.md                     # This file <br/>

**Notes**

The model uses a fixed random seed (2147483647) for reproducibility.
Training is performed with a single epoch in the provided code, but you can increase the number of iterations for better results.
The loss function includes L2 regularization (0.01*(W**2).mean()) to prevent overfitting.
The notebook includes visualizations (e.g., bigram frequency matrix) using matplotlib.
<br/>
**License**
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
<br/>
Inspired by Andrej Karpathy's "Makemore" series on building neural networks for language modeling.
Built with PyTorch and Jupyter Notebook.

