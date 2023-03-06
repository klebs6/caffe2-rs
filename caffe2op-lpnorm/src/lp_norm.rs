crate::ix!();

/**
  | This op computes the $L_p$ norm of the
  | one dimensional input tensor $X$, and
  | outputs a one dimensional output tensor
  | $Y$. Here, the $L_p$ norm is calculated
  | as
  | 
  | $$L_p(\mathbf{x}) = \sum_i x_i^p$$
  | 
  | This op supports $p$ values of 1 or 2.
  | If the average argument is set, the norm
  | is calculated as
  | 
  | Lp_averaged_norm(x) is defined as
  | 
  | Lp_averaged_norm(x) = LpNorm(x) /
  | size(x).
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.h
  | //- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LpNormOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    p:       i32,
    average: bool,
    phantom: PhantomData<T>,
}

num_inputs!{LpNorm, 1}

num_outputs!{LpNorm, 1}

inputs!{LpNorm, 
    0 => ("X", "1D Input tensor of data to be operated on.")
}

outputs!{LpNorm, 
    0 => ("Z", "1D output tensor")
}

args!{LpNorm, 
    0 => ("p",       "*(type: int; default: 2, possible values: {1,2})* Order of the norm in p-norm."),
    1 => ("average", "*(type: bool; default: False)* Whether we calculate norm or averaged_norm.The Lp_averaged_norm(x) is defined as Lp_averaged_norm(x) = LpNorm(x) / size(x)")
}

tensor_inference_function!{LpNorm, /* [](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      std::vector<int64_t> output_dims(1);
      output_dims[0] = 1; // 1
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    } */
}

impl<T, Context> LpNormOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "p", p_, 2),
            OP_SINGLE_ARG(bool, "average", average_, false) 

        CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
        */
    }
}
