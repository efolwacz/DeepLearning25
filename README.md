# DeepLearning25
Material zur Vorlesung + Labor Deep Learning an der HSO im SS25

## Hinweise
* Verwenden Sie Ihr [*HSO Google* account](https://hilfe.cit.hs-offenburg.de/confluence/citpublic/google-workspace-hilfeseiten) um die Laboraufgaben in [*Colab*](https://colab.research.google.com) auszuführen.
* Speichern Sie ihre Ergebnisse/Code in *Colab* über *download*, *save to google drive* 

<details>
<summary> <H2> Woche 1 - Intro </H2><BR></summary>

### Colab Intro
* [Colab Tutorial](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

#### Advanced Colab Topics (optinal)
* [Markdown in Colab](https://colab.research.google.com/notebooks/markdown_guide.ipynb)
* [Accessing Data from Colab](https://colab.research.google.com/notebooks/io.ipynb)
* [Colab GitHub Integration](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)


### Aufgabe 1
* [CIFAR10 Challenge](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_1/CIFAR10-ShallowLearning.ipynb)

</details>

<details>
<summary> <H2> Woche 2 - From Perceptron to MLP Networks </H2><BR></summary>

### Demos
* [Demo: single neuron](https://playground.tensorflow.org/#activation=linear&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.29245&showTestData=true&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
* [Demo: single neuron - multi class](https://playground.tensorflow.org/#activation=linear&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.34827&showTestData=true&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

### NN from scratch in Python
* [Single Neuron](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_2/A_simple_Perceptron_in_NumPy.ipynb)

### Aufgabe 2
* [Multi Class Perceptron](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_2/Aufgabe_2_Multi_Class_Perceptrons.ipynb) -> [solution](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_2/Multi_Class_Perceptrons_solution.ipynb)

</details>

<details>
<summary> <H2> Woche 3 - Training NNs I </H2><BR></summary>

### Vorlesung
* [Training a simple Perceptron](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_3/Training%20_a_simple_Perceptron_in_NumPy.ipynb)
* [Training Video](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_3/train_video.gif)

### Lab
* [Intro PyTorch tensors](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_3/pytorch_tensors.ipynb) 
* [Perceptron in PyTorch](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_3/a_perceptron_in_PyTorch.ipynb)

### PyTorch
* [Tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)
* [API](https://pytorch.org/docs/stable/index.html)

### Aufgabe 3
* [Assignment: MLP in Pytorch](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_3/Assignment_Basic_MLP_in_Pytorch.ipynb) -> [solution](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_3/Assignment_Basic_MLP_in_Pytorch_solution.ipynb)

</details>
<details>
<summary> <H2> Woche 4 - Training NNs II </H2><BR></summary>

### Lab
* [Data Loader and GPU usage](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_4/PyTorch_DataLoderandGPU.ipynb)
* [TensorBoard with PyTorch on Colab tutorial](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_4/tensorboard_with_pytorch.ipynb)
* [PyTorch AutoGrad](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_4/autograd_tutorial.ipynb)

### Aufgabe 4
* [Assignment: Optimizing and Analyzing NN Training](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_4/Assignment_CIFAR10_MLP_optimization.ipynb) -> [solution](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_4/Assignment_CIFAR10_MLP_optimization_solution.ipynb)

</details>
<details>
<summary> <H2> Woche 5 - CNNs  </H2><BR></summary>
  
### Lab
* [AlexNet Implementation in PyTorch](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/3_alexnet.ipynb)

### Get State of the Art Models: 
* [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html)
* [Papers with Code](https://paperswithcode.com/sota)
* [Hugging Face Models](https://huggingface.co/models)

### Aufgabe 5
* [Assignment: PyTorch Model Zoo](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_5/Assignment_PyTorch_Model_Zoo.ipynb) -> [solution](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_5/Assignment_PyTorch_Model_Zoo_solution.ipynb)


</details>

<details>
<summary> <H2>Woche 6 - RNNs </H2><BR></summary>


### Lab
* [LSTMs with PyTorch](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_6/Lab_Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb) 

### Assignments
* [Stock Price Prediction](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_6/Assignemnt_stock-price.ipynb)


</details>

<details>
<summary> <H2>Woche 7 - Transformers </H2><BR></summary>


### Lab
* [Using a pre-trained Vision transformer](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_7/pre_trained_vt.ipynb)
* [fine tuning a pre-trained transformer](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_7/fine_tune_pre_trained_transformer.ipynb)

### Assignments
* [Transformer on CIFAR10](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_7/Transformer_CIFAR10.ipynb) -> [solution](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_7/Transformer_CIFAR10_solution.ipynb)


</details>

<details>
<summary> <H2>Woche 8 - Semi-, Self- and Un-Supervised Training | Autoencoders </H2><BR></summary>
  
### Links:
* [PyTorch augmentation transformations](https://pytorch.org/vision/stable/transforms.html)
* [PyTorch AutoAugmentation](https://pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html)

### Assignent
* [AutoEncoder on MNIST](https://colab.research.google.com/github/keuperj/DeepLearning25/blob/main/week_8/Assignment_AE_MNIST.ipynb) 
</details>


