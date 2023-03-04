crate::ix!();

/**
  | Hierarchical softmax is an operator
  | which approximates the softmax operator
  | while giving significant training
  | speed gains and reasonably comparable
  | performance. In this operator, instead
  | of calculating the probabilities of
  | all the classes, we calculate the probability
  | of each step in the path from root to the
  | target word in the hierarchy.
  | 
  | The operator takes a 2-D tensor (Tensor)
  | containing a batch of layers, a set of
  | parameters represented by the weight
  | matrix and bias terms, and a 1-D tensor
  | (Tensor) holding labels, or the indices
  | of the target class. The hierarchy has
  | to be specified as an argument to the
  | operator.
  | 
  | The operator returns a 1-D tensor holding
  | the computed log probability of the
  | target class and a 2-D tensor of intermediate
  | outputs (from the weight matrix and
  | softmax from each step in the path from
  | root to target class) which will be used
  | by the gradient operator to compute
  | gradients for all samples in the batch.
  |
  */
pub struct HSoftmaxOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base:    HSoftmaxOpBase<T, Context>,

    phantom: PhantomData<T>,
}

num_inputs!{HSoftmax, 4}

num_outputs!{HSoftmax, 2}

inputs!{HSoftmax, 
    0 => ("X",                   "Input data from previous layer"),
    1 => ("W",                   "2D blob containing 'stacked' fully connected weight matrices. Each node in the hierarchy contributes one FC weight matrix if it has children nodes. Dimension is N*D, D is input dimension of data (X), N is sum of all output dimensions, or total number of nodes (excl root)"),
    2 => ("b",                   "1D blob with N parameters"),
    3 => ("labels",              "int word_id of the target word")
}

outputs!{HSoftmax, 
    0 => ("Y",                   "1-D of log probability outputs, one per sample"),
    1 => ("intermediate_output", "Extra blob to store the intermediate FC and softmax outputs for each node in the hierarchical path of a word. The outputs from samples are stored in consecutive blocks in the forward pass and are used in reverse order in the backward gradientOp pass")
}

args!{HSoftmax, 
    0 => ("hierarchy",           "Serialized HierarchyProto string containing list of vocabulary words and their paths from root of hierarchy to the leaf")
}
