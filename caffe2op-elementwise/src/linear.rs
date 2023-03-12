crate::ix!();

/**
  | This op computes the elementwise linear
  | combination of a batch of input vectors
  | with a weight vector and bias vector.
  | As input, the op takes an input tensor
  | $X$ of shape $NxD$, a weight vector $w$
  | of length $D$, and a bias vector $b$ of
  | length $D$.
  | 
  | Here, $N$ represents the batch size
  | and $D$ represents the length of the
  | feature vectors. The output, $Y$, is
  | a tensor of shape $NxD$ and is calculated
  | as
  | 
  | $$Y_{ij} = X_{ij}w_j + b_j \ for \ i\in{N},
  | j\in{D}$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ElementwiseLinearOp<T, Context, Engine> {
    storage:  OperatorStorage,
    context:  Context,
    axis:     i32,
    phantom:  PhantomData<T>,
    phantomE: PhantomData<Engine>,
}

num_inputs!{ElementwiseLinear, 3}

num_outputs!{ElementwiseLinear, 1}

inputs!{ElementwiseLinear, 
    0 => ("X", "2D input tensor of size $NxD$. This input represents the input data to be operated on."),
    1 => ("w", "1D scaling factors, or weights, of size $D$. This input contains the weights that will be multiplied by the data."),
    2 => ("b", "1D biases of size $D$. This input contains the biases that will be added to the products of the weights and data.")
}

outputs!{ElementwiseLinear, 
    0 => ("Y", "2D output tensor of size $NxD$. Calculated as described above.")
}

args!{ElementwiseLinear, 
    0 => ("axis", "*(type: int; default: 1)* Describes the axis of the inputs; defaults to one because the 0th axis most likely describes the batch size.")
}

inherit_onnx_schema!{ElementwiseLinear}

register_cpu_operator!{
  ElementwiseLinear,
  ElementwiseLinearOp<f32, CPUContext>
}

impl<T,Context,Engine> ElementwiseLinearOp<T,Context,Engine> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
}
