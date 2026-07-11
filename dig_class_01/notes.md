### Why dataset.py and train.py are different files?

This is in order to seperate the concern of loading, preprocessing and preparing the data and that of actually training the weights of the the neural net using forward prop, backward prop and gradient descent. \
So, in future if I had to increase the data, modify it to include more samples (to improve the network in general) I would do so without touching the training plane at all. \
Similarly, on the other hand if I need to test and experiment different learning algorithms for performance I can do so easily by modifying the train.py.

### What information is lost during flattening? Can we retrieve it back?

Dimensionality information is lost during flattening. But all the feature based information is conserved. We might be able to retrieve the lost dimensionality information if we store it before hand and then use it to reconstruct our original input data (before flattening).

### What can the 10 output neurons?

They can represent the following things:

1. Either binary output values that would then represent the digits from 0-9. For example, the output [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] represents the fact that the input image is the digit '7',
2. They can also represent probability values representing the likelihood of an input image being a particular digit. For example, the index of the neuron having the largest probability value, would tell us what input digit it was,
3. Thirdly, the output layer could also represent 10 different, but richer representation of the input layer. Richer in the sense of the characteristics of the digits; an example being [ number of horizontal strokes, number of vertical strokes, amount of curvature, presence of a closed loop, brightness in the top half, brightness in the bottom half,
   left-right symmetry, center of mass, stroke thickness, overall ink density ] (The main aim of the third hypothesis is to understand that every layer does not necessarily give theoutput representation; many hidden layers convert input to a different yet more understandable and learnable representation by performing transformations on vectors using matrices).

### Design Document of a neuron

A neuron should contain the following things:

1. The weight (w) associated with it,
2. It should accept the output from the previous layer as input (x),
3. Additionally it may also contain a bias term associated with it (b)

### Should every weight initially be 0?

If every weight is initially 0, the output of the neuron would be initially 0. This might not give a good starting point for our learning algorithm to understand the task at hand. In better words, if all the weights start at zero, every neuron in a hidden layer will perform the exact same calculations. So basically every neuron would be a clond of each other and the complete network will effectively act as a single neuron. This is also called as the **Symmetry Problem**. It leads to the network being unable to learn complex and varied features from the input vector. \
Instead, if the weights are assigned random values intially, this would mean that intrinsically we are providing an output for the neuron (whether a good one or a bad one). This output would then be interpreted in a mathemtically better way by the particular learning algorithm to proceed further with understanding the task at hand.

### What role does bias play?

Bias plays the role of defining the intrinsic activation value of a neuron. In fact, during the start of the learning phase, it is best to set the bias to 0. And then as learning occurs and the importance of different neurons vary, bias is changed and fine-tuned. \
To give a more intuitive explanation, think of bias as the neuron's tendency to activate. Basically answering the question: how likely the input is going to negatively or positively affect the output of a neuron ? This question is answered by the value of the bias. \
