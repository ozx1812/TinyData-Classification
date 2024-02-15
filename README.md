# Image Classification with Minimal Data: Innovations in Few-Shot Learning and Unsupervised Feature Extraction

## Models Explored

### EvoDCNN

Utilizes Genetic Algorithm (GA) to evolve hyperparameters of a Deep Convolutional Neural Network (DCNN). Improved accuracy scores demonstrated across various datasets, including CIFAR10, MNIST, and EMNIST.

### Relational Networks (RNs)

Introduces a novel architecture based on RNs for few-shot learning. RNs utilize learnable transformation functions to compare input examples and recognize patterns. Evaluated on benchmarks such as Omniglot and mini-ImageNet, showcasing competitive performance against other few-shot learning methods.

### MAML (Model Agnostic Meta Learning)

Learns a good initialization of model parameters adaptable to new tasks with minimal examples. Explores the effects of model depth and linear layers on meta-learning performance. Tested on datasets like Omniglot, CIFAR-FS, and mini-Imagenet, showing improved performance with increased depth and novel preconditioning techniques.

### SWaV (Unsupervised Learning of Visual Features by Contrasting Cluster Assignments)

Presents an unsupervised learning approach for visual feature learning utilizing clustering, optimal transport, and sinkhorn’s theorem. Online clustering and soft clustering techniques utilized to handle large datasets efficiently. Advantages over other self-supervised learning approaches discussed, highlighting its potential in downstream tasks like classification.

The significance of few-shot learning lies in its ability to learn from only a limited amount of data, which makes it a critical area of research in the field of machine learning. In this literature review, we will explore models that can be adapted for few-shot learning, as well as models that can incorporate external data to enhance the performance of few-shot learning.

## EvoDCNN

[Sun et al., 2019](https://arxiv.org/abs/1710.10741)

Developing a Deep Convolutional Neural Network (DCNN) for image classification is a very dataset-specific task as an effective network will be tuned with respect to the dataset that it is used on. Tuning the hyperparameters of a model is a time and resource-consuming matter. This paper presents the implementation of Genetic Algorithm (GA) to developing a DCNN. The GA is used to evolve the network and find the best hyperparameters. 17 parameters are being tuned across 3 blocks. The convolution block contains hyperparameters: number of blocks, number of convolution layers, filter size, number of filters, dropout, pooling, activation function, short connection, and batch normalization. The classification block contains hyperparameters: type of layer, number of nodes, batch normalization, activation function, and dropout. The training block contains the optimizer, learning rate, batch normalization, and initialization. 

The GA contains four steps: initialization, selection, crossover, and mutation. During the initialization, the hyperparameters for the networks are selected randomly. The Roulette wheel selection model is then used to select networks that will be used for crossover and mutation. Crossover mixes hyperparameters from two or more networks to create new networks. Mutation creates a random change in one or more of the hyperparameters of a model. This evolution can continue for as many generations as needed, aiming to find the best hyperparameters.

The model is evaluated on eight datasets including CIFAR10, MNIST, and six versions of EMNIST. All datasets showed an improvement in their accuracy scores as the number of generations increased, with five having error rates lower than the state-of-the-art.

## Relational Networks

[Sung et al., 2018](https://arxiv.org/abs/1711.06025)

The field of few-shot learning has gained significant attention in the research community in recent years, as it aims to solve one of the biggest issues in the domain of machine learning, which is the unavailability of huge datasets required by traditional machine learning and deep learning models. In this paper, the authors propose a new approach to few-shot learning based on Relation Networks (RNs). The authors introduce a new architecture and compare it against several methods on a variety of datasets.

The paper proposes a novel architecture called Relation Networks (RNs) for few-shot learning. RNs utilize a set of learnable transformation functions that take pairs of feature vectors as inputs and output a score representing the relationship between them. These functions can be thought of as a learnable similarity metric, allowing the network to compare different input examples and learn to recognize patterns. The RN architecture consists of multiple stages, each of which applies a set of these transformation functions to the input features. The final output of the network is a classification score based on the transformed features.

The authors evaluate the effectiveness of their proposed method on a range of few-shot learning benchmarks, including Omniglot, mini-ImageNet. They compare RNs against several other few-shot learning methods, including Matching Networks, and Prototypical Networks.

## MAML

[Arnold et al., 2021](https://arxiv.org/abs/1910.13603)

Model Agnostic Meta Learning (MAML) aims to learn a good initialization of model parameters that can be quickly adapted to new tasks with only a few examples. This paper answers three main questions regarding MAML: (1) What is the effect of the depth of a model? (2) What is the role of the model's heads in updating the bottom layers of an MAML? (3) How do the additional parameters introduced by preconditioning methods help the original method that does not contain these parameters?

The depth of a model is deemed important as its upper layers to control the lower layers’ parameter updates. This can be done by increasing the depth of the network, adding layers at the output of the model, or using preconditioning methods. To explore the effects of a model’s depth and linear layers, a baseline of a CNN with 4 convolutional layers is considered. This model is considered a failure scenario, meaning that it will fail to meta-learn. The depth of the model is then increased until the model becomes meta-learnable. The same principle is applied to the number of linear layers applied to the output. Layers are added to the failure scenario until it becomes meta-learnable.

Preconditioning techniques aimed at transforming models’ gradients, such as First-Order Approximation are also known to make models more meta-learnable. This paper’s take on preconditioning is the algorithm META-KFO. This algorithm consists of a neural network that learns to transform the gradient on models that would otherwise not be able to meta-learn, without changing the base of the model’s modeling capacity.

META-KFO is tested along with MAML on Omniglot, CIFAR-FS, and mini-Imagenet datasets and shows better performance than other meta-optimizers on both the Omniglot and CIFAR-FS datasets. However, as the number of layers of a model is increased, the accuracy score of the MAML model catches up with the MAML+KFO model. This suggests that increasing the depth has a similar effect to adding a meta-optimizer on the meta-learning of a model.

## SWaV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

[Caron et al., 2021](https://arxiv.org/abs/2006.09882)

### Introduction

Self-supervised learning has become a popular technique for learning visual features without the need for labeled data. SWaV is a novel unsupervised learning approach for visual feature learning that utilizes clustering, optimal transport, and sinkhorn's theorem. In this review, we discuss the SWaV approach and its advantages over other self-supervised learning techniques.

### Approach

SWaV replaces the supervised pre-training of CNN with unsupervised CNN by leveraging the property of images. The property of images that is used is that semantic information is contained within an image, as well as its different versions, distortions, and crops. The approach uses clustering to predict the group assignment of a distortion from another distortion of the same image.

### Clustering

Clustering is used in SWaV because computing all pairwise comparisons on a large dataset is not practical, and most implementations approximate the loss by reducing the number of comparisons to random subsets. However, to cluster images, a full forward pass on the entire dataset is needed, which is intractable. Therefore, the approach uses online clustering to solve this problem.

### Optimal Transport

The SWaV approach enforces the constraint that all prototypes should be equally representative. Without this constraint, the network would cheat and assign all features to one prototype, making it too easy to predict the prototype. The problem of assigning features to prototypes can be solved using the sinkhorn-knopp algorithm in optimal transport. The solution in the algorithm is an ascendant matrix, which assigns a feature to a vector of probabilities of belonging to each prototype.

### Soft Clustering

The vector of probabilities is then rounded using the highest probability in the vector, which is called soft clustering. Sinkhorn-knopp algorithm works on the basis of a matrix A (n x n) with strictly positive elements. There exist two diagonal matrices D1 and D2 with strictly positive elements such that D1 x A x D2 becomes doubly stochastic.

### Multi-crop

The SWaV approach also utilizes the idea of multi-crop, which came from the self-supervised learning of pretext-invariant representation. The idea is that the improvement in the SWaV approach was not due to the jigsaw puzzle task, but rather due to the data augmentations used in the jigsaw objective. The approach aims to do what has been done in supervised augmentation but needs to be done in unsupervised augmentation.

### Comparison with other approaches

SWaV has advantages over other self-supervised learning approaches. Cluster-based approaches are generally offline and require a full pass on the data. Methods based on noise contrastive estimation generally calculate contrastive loss using strong data augmentation techniques. These techniques become computationally expensive for large datasets. Additionally, these models maintain a large memory bank, which introduces computational expense.

### Conclusion

In conclusion, SWaV is a novel approach for self-supervised visual feature learning that utilizes clustering, optimal transport, and sinkhorn's theorem. The approach is advantageous over other self-supervised learning approaches as it is online, does not require strong data augmentation techniques, and does not maintain a large memory bank. The approach could potentially be useful in downstream tasks such as classification, where inter-image similarities are important.
