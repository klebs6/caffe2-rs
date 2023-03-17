This rust crate provides mathematical operators
for DSP and machine learning computations.

**Note: This crate is currently being translated from C++ to Rust, and some function bodies may still be in the process of translation.**

It includes operations for implementing activation
functions commonly used in neural networks and
other machine learning models. The crate provides
functions for computing the forward and backward
pass of these activation functions, such as
cudnnActivationForward and
cudnnActivationBackward.

The crate also provides functions for setting and
modifying the activation function properties, such
as the activation mode, which is defined by
CudnnActivationMode and kCudnnActivationMode. The
SetTensorDescriptor function is also included for
setting up the input and output tensors.

The crate implements various activation functions,
such as the sigmoid function, which is defined by
the equation:

f(x) = 1 / (1 + e^(-x))

Another example is the ReLU function, which is
defined as:

f(x) = max(0, x)

Overall, this rust crate is a useful tool for
implementing and optimizing machine learning
models that rely on activation functions, and for
performing DSP computations in general.

3> would you please list for me as many activation
functions as you can think of along with their
mathematical representations?

Certainly, here are some commonly used activation
functions in machine learning along with their
mathematical representations:

### 1. Sigmoid Function: 

f(x) = 1 / (1 + e^(-x))

The sigmoid function is an activation function
that takes a real-valued number as input and
returns a value between 0 and 1. It is defined by
the equation f(x) = 1 / (1 + e^(-x)). The function
has a characteristic S-shaped curve, and its
output is used to represent a probability-like
value for the input. The function is commonly used
in binary classification tasks, where the output
represents the probability of belonging to one of
two classes.

### 2. Rectified Linear Unit (ReLU) Function: 

f(x) = max(0, x)

The ReLU function is an activation function that
sets all negative values of the input to zero and
leaves all non-negative values unchanged. It is
defined by the equation f(x) = max(0, x). This
function is commonly used in deep neural networks
because it is computationally efficient and does
not suffer from the "vanishing gradient" problem
that other activation functions can experience.

### 3. Leaky ReLU Function: 

f(x) = max(0.01x, x)

The leaky ReLU function is similar to the ReLU
function but with a small slope for negative
values. It is defined by the equation f(x)
= max(0.01x, x). This function is used to overcome
the "dying ReLU" problem that can occur when the
ReLU function outputs zero for a large number of
inputs.

### 4. Parametric ReLU Function: 

f(x) = max(alpha*x, x), 
where alpha is a learnable parameter

The parametric ReLU function is a variant of the
ReLU function that includes a learnable parameter
alpha. It is defined by the equation f(x)
= max(alpha*x, x), where alpha is a learnable
parameter. This function can help improve the
performance of deep neural networks.

### 5. Exponential Linear Unit (ELU) Function: 

f(x) = x if x >= 0, 
f(x) = alpha*(e^x - 1) if x < 0, 
where alpha is a hyperparameter

The ELU function is similar to the ReLU function
but with a smoother transition around zero. It is
defined by the equation f(x) = x if x >= 0, f(x)
= alpha*(e^x - 1) if x < 0, where alpha is
a hyperparameter. This function is designed to
address the vanishing gradient problem and can
improve the performance of deep neural networks.

### 6. Softplus Function: 

f(x) = log(1 + e^x)

The softplus function is a smooth approximation of
the ReLU function. It is defined by the equation
f(x) = log(1 + e^x). 

This function is used in deep neural networks as
an alternative to the ReLU function because it is
differentiable at zero and provides a smoother
gradient.

### 7. Hyperbolic Tangent (Tanh) Function: 

f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

The hyperbolic tangent function is an activation
function that maps the input to a value between -1
and 1. 

It is defined by the equation 
f(x) = (e^x - e^(-x)) / (e^x + e^(-x)). 

This function is commonly used in recurrent neural
networks (RNNs) because it is sensitive to the
magnitude of the input and can be used to control
the flow of information through the network.

### 8. Swish Function: 
f(x) = x * sigmoid(x)

The Swish function is a relatively new activation
function that is similar to the sigmoid function
but has a learnable parameter beta. It is defined
by the equation f(x) = x * sigmoid(beta*x). This
function has been shown to outperform other
activation functions on a variety of deep learning
tasks.

## Note
There are many other activation functions that can
be used depending on the problem at hand, but
these are some of the most commonly used ones.
