## caffe2op-crossentropy

This Rust crate implements a collection of
mathematical operators and functions commonly used
in machine learning and deep learning applications
for calculating cross-entropy loss. Cross-entropy
loss is a popular loss function used in
classification tasks to evaluate the difference
between predicted class probabilities and true
class labels. The crate provides different
versions of cross-entropy loss functions,
including `CrossEntropyOp`, `SoftMax`,
`SigmoidCrossEntropyWithLogitsOp`, and
`WeightedSigmoidCrossEntropyWithLogitsOp`, among
others.

The crate also provides gradient computation
functions for each of these loss functions,
including `CrossEntropyGradientOp`,
`SigmoidCrossEntropyWithLogitsGradientOp`, and
`WeightedSigmoidCrossEntropyWithLogitsGradientOp`. The
`MakeTwoClassOp` and `MakeTwoClassGradientOp`
functions provide an implementation of the
Goodfellow et al. method for creating a binary
classification task from a multi-class
classification problem.

The crate provides a range of useful functions and
operators such as `Soft`, `Collect`,
`ResetWorkspace`, `FeedBlob`, and `FetchBlob` for
managing the workspace and data flow in deep
learning applications. The crate also offers
functions such as `GetCrossEntropyGradient`,
`GetLabelCrossEntropyGradient`, and
`GetWeightedSigmoidCrossEntropyWithLogitsGradient`,
which return the gradient of a given loss function
with respect to the input data.

This crate's implementation of cross-entropy loss
functions is stable, ensuring numerical stability
when computing cross-entropy loss to avoid
vanishing and exploding gradients. The crate is
well-documented and supported, and provides
functions and operators for a range of machine
learning applications.

### Relevant Mathematical Ideas

- **Cross-entropy loss**: A loss function commonly
  used in classification tasks to evaluate the
  difference between predicted class probabilities
  and true class labels. The cross-entropy loss is
  defined as follows:

  `H(p, q) = -∑[i=1 to n]p(i)log(q(i))`

  where `p` is the true class distribution, `q` is
  the predicted class distribution, and `n` is the
  number of classes.

- **Softmax**: An activation function that
  converts a vector of real numbers to
  a probability distribution over the classes. The
  softmax function is defined as follows:

  `softmax(x) = exp(x) / ∑[j=1 to K]exp(x(j))`

  where `K` is the number of classes.

- **Sigmoid cross-entropy with logits**: A loss
  function used for binary classification
  tasks. The sigmoid cross-entropy with logits is
  defined as follows:

  `sigmoid_cross_entropy_with_logits(x, t) 
  = max(x, 0) - x*t + log(1 + exp(-abs(x)))`

  where `x` is the predicted output, `t` is the
  true label, and `abs(x)` is the absolute value
  of `x`.

- **Weighted sigmoid cross-entropy with logits**:
  A variant of the sigmoid cross-entropy with
  logits loss function that allows for class
  weighting. The weighted sigmoid cross-entropy
  with logits is defined as follows:

  `weighted_sigmoid_cross_entropy_with_logits(x, t, pos_weight, neg_weight) 
  = pos_weight * (max(x, 0) - x*t) + neg_weight * log(1 + exp(-max(x, 0)))`

  where `x` is the predicted output, `t` is the
  true label, `pos_weight` is the weight for
  positive examples, and `neg_weight` is the
  weight for negative examples.
