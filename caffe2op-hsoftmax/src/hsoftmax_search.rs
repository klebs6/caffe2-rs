crate::ix!();

/**
  | HSoftmaxSearch is an operator to generate
  | the most possible paths given a well-trained
  | model and input vector. Greedy algorithm
  | is used for pruning the search tree.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HSoftmaxSearchOp<T, Context> {
    base:    HSoftmaxOp<T, Context>,
    top_n:   i32,
    beam:    f32,
    tree:    TreeProto,
    phantom: PhantomData<T>,
}

num_inputs!{HSoftmaxSearch, 3}

num_outputs!{HSoftmaxSearch, 2}

inputs!{HSoftmaxSearch, 
    0 => ("X", "Input data from previous layer"),
    1 => ("W", "The matrix trained from Softmax Ops"),
    2 => ("b", "The bias trained from Softmax Ops")
}

outputs!{HSoftmaxSearch, 
    0 => ("Y_names", "The name of selected nodes and leafs. For nodes, it will be the name defined in the tree. For leafs, it will be the index of the word in the tree."),
    1 => ("Y_scores", "The corresponding scores of Y_names")
}

args!{HSoftmaxSearch, 
    0 => ("tree", "Serialized TreeProto string containing a tree including all intermidate nodes and leafs. All nodes must have names for correct outputs"),
    1 => ("beam", "beam used for pruning tree. The pruning algorithm is that only children, whose score is smaller than parent's score puls beam, will be propagated. "),
    2 => ("topN", "Number of nodes in outputs")
}

should_not_do_gradient!{HSoftmaxSearch}

impl<T, Context> HSoftmaxSearchOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : HSoftmaxOp<T, Context>(std::forward<Args>(args)...),
            top_n_(this->template GetSingleArgument<int>("topN", 5)),
            beam_(this->template GetSingleArgument<float>("beam", 0.01f)) 

        CAFFE_ENFORCE(tree_.ParseFromString(
            this->template GetSingleArgument<string>("tree", "")));
        */
    }
}
