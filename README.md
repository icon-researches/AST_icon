# Adatpive Synaptic Template (AST)
Spiking Neural Networks (SNNs) classifier files with Adaptive Synaptic Template

# Descritption
These files are main files for classifying data samples with Spiking Neural Networks (SNNs) based on Diehl&Cook2015.
I built these files with BindsNET. 
Therefore, they demand BindsNET to operate.
To use AST, they require our add-on packages with additional features that I have developed.
Unfortunately, I do not have permission to upload the packages on Github.
Please install BindsNET and contact me to get the add-on packages.

# How to use AST with additional package
I cannot upload our add-on packages because I do not have permission to release.
If you need the package and more datasets, please e-mail me. (hetzer44@naver.com)
Then, I will send the package and datasets.
For using the package, just put the package into the bindsnet folder in the python library folder (site-packages)

# Classifiers
snn_wq.py is a Python file for classifying Wine Quality (WQ) dataset.

snn_digits.py is a Python file for classifying MNIST (8x8 pixels) dataset.

ast_minst.py is a Python file for classifying MNIST (28x28 pixels) dataset.

ast_fminst.py is a Python file for classifying Fashion MNIST (28x28 pixels) dataset.

Python files with 'batch' update synaptic weights using batches.

# Implementation
We implemented two versions of AST. 
One is implemented by using conditions and loops. 
The other is implementde by using pytorch methods.
They show the same results but they have slightly different operations.

# Paper
My paper explaining AST is under review on Wiley Computational Intelligence!
The preprint is in here.
https://www.researchgate.net/publication/374136675_Adaptive_synaptic_adjustment_mechanism_to_improve_learning_performances_of_spiking_neural_networks
