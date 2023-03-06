crate::ix!();

/**
  | Matrix multiplication $Y = A * B$, where
  | `A` has size (M x K), `B` has size (K x N),
  | and `Y` will have a size (M x N).
  | 
  | To transpose `A` or `B` before multiplication,
  | pass 1 to the `trans_a` and/or `trans_b`
  | arguments, which separate the first
  | and second dimensions of the respective
  | matrices using `axis_a` and `axis_b`.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/matmul_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MatMulOp<T, Context, Engine> {

    storage: OperatorStorage,
    context: Context,

    /**
      | A local vector to cache the output shape
      | so we don't need to recreate a vector
      | object every time we run Run().
      |
      */
    y_shape_cache: Vec<i64>, // default = {0, 0};

    axis_a:   i32, // default = 1
    axis_b:   i32, // default = 1
    trans_a:  bool,
    trans_b:  bool,
    phantom: PhantomData<T>,
    phantomE: PhantomData<Engine>,
}

num_inputs!{MatMul, (2,3)}

num_outputs!{MatMul, 1}

inputs!{MatMul, 
    0 => ("A", "*(type: Tensor`<float>`)* 2D matrix of size (M x K)."),
    1 => ("B", "*(type: Tensor`<float>`)* 2D matrix of size (K x N).")
}

outputs!{MatMul, 
    0 => ("Y", "*(type: Tensor`<float>`)* 2D matrix of size (M x N).")
}

args!{MatMul, 
    0 => ("axis_a", "*(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `A`."),
    1 => ("axis_b", "*(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `B`."),
    2 => ("trans_a", "*(type: int; default: 0)* Pass 1 to transpose `A` before multiplication and after the dimension adjustment using `axis_a`."),
    3 => ("trans_b", "*(type: int; default: 0)* Pass 1 to transpose `B` before multiplication and after the dimension adjustment using `axis_b`.")
}

register_cpu_operator!{MatMul, MatMulOp<f32, CPUContext>}

register_cuda_operator!{MatMul, MatMulOp<float, CUDAContext>}
