crate::ix!();

/**
  | The FC operator computes an output $(Y)$
  | as a linear combination of the input
  | data blob $(X)$ with a weight blob $(W)$
  | and bias blob $(b)$. More formally,
  | 
  | $$Y = XW^T+b$$
  | 
  | Here, $X$ is a matrix of shape $(M,K)$,
  | $W$ is a matrix of shape $(N,K)$, $b$
  | is a vector of length $N$, and $Y$ is a
  | matrix of shape $(M,N)$. $N$ can be thought
  | of as the number of nodes in the layer,
  | $M$ is the batch size, and $K$ is the number
  | of features in an input observation.
  | 
  | -----------
  | @note
  | 
  | $X$ does not need to explicitly be a 2-dimensional
  | matrix, however, if it is not it will
  | be coerced into one. For an arbitrary
  | $n$-dimensional tensor $X$, e.g. $[a_0,
  | a_1, \ldots ,a_{k-1}, a_k, \ldots ,
  | a_{n-1}]$, where $a_i$ in $N$, and $k$
  | is the $axis$ arg provided, then $X$
  | will be coerced into a 2-dimensional
  | tensor with dimensions $[a_0 * \ldots
  | * a_{k-1}, a_k * \ldots * a_{n-1}]$.
  | For the default case where axis=1, this
  | means the $X$ tensor will be coerced
  | into a 2D tensor of dimensions $[a_0,
  | a_1 \ldots * a_{n-1}]$, where $a_0$
  | is often the batch size. In this situation,
  | we must have $a_0 = M$ and $a_1 * \ldots
  | * a_{n-1} = K$. Lastly, even though $b$
  | is a vector of length $N$, it is copied
  | and resized to shape $(M x N)$ implicitly,
  | then added to each vector in the batch.*
  | 
  | This is Caffe's InnerProductOp, with
  | a name that fits its purpose better.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FullyConnectedOp<Context, Engine, const TransposeWeight: bool> {
    storage:         OperatorStorage,
    context:         Context,
    axis:            usize, //{1};
    axis_w:          usize, //{1};

    /**
      | A local vector to cache the output shape
      | so we don't need to recreate a vector
      | object every time we run Run().
      |
      */
    y_shape_cache:   Vec<i64>,
    bias_multiplier: Option<Tensor>,
    float16_compute: bool,
    phantomE:        PhantomData<Engine>,
}

num_inputs!{FC, 3}

num_outputs!{FC, 1}

inputs!{FC, 
    0 => ("X", "Input blob to be coerced into a 2D matrix of shape $(M,K)$, where $M$ is the batch size and $K$ is the number of features in a single observation."),
    1 => ("W", "Input blob to be coerced into a 2D matrix of shape $(N,K)$ describing a fully connected weight matrix. Here, $K$ is the number of features in a single observation and $N$ is the number of nodes in the FC layer."),
    2 => ("b", "Input blob containing vector of length $N$ which describes one bias for each node in the layer.")
}

outputs!{FC, 
    0 => ("Y", "Output blob containing a 2D output matrix of shape $(M,N)$, where $M$ is the batch size and $N$ is the number of nodes in the layer. The output is calculated as $Y=XW^T+b$.")
}

args!{FC, 
    0 => ("axis", "*(type: int; default: 1)* Describes the axis of the input data $X$. Defaults to one because in the common case when the input $X$ has shape $(M,K)$, the first axis encodes the batch size."),
    1 => ("axis_w", "*(type: int; default: 1)* Describes the axis of the input weight matrix $W$. Defaults to one because the first axis most likely describes the batch_size."),
    2 => ("float16_compute", "*(type: bool; default: False)* Whether to use float-16 compute kernel.")
}

inherit_onnx_schema!{FC, "Gemm"}

tensor_inference_function!{FC, /* std::bind(FCShapeInference, _1, _2, false) */}

cost_inference_function!{FC, /* std::bind(CostInferenceForFC, _1, _2, false) */}

register_cpu_operator!{
    FC, 
    FullyConnectedOp<CPUContext>
}
