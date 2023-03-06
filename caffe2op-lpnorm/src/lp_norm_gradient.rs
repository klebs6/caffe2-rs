crate::ix!();

/**
  | Given one input float tensor X, derivative
  | dout, and produces one output float
  | tensor dX.
  | 
  | dX is the derivative of the Lp norm of
  | tensor X, computed as dx = d(sum over
  | |x^p|)/dx, in which p is either 1 or 2(currently
  | only supports l1 and l2 norm) determined
  | by the argument p.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LpNormGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    p:       i32,
    average: bool,
    phantom: PhantomData<T>,
}

num_inputs!{LpNormGradient, 2}

num_outputs!{LpNormGradient, 1}

inputs!{LpNormGradient, 
    0 => ("X", "1D input tensor"),
    1 => ("dout", "1D input tensor")
}

outputs!{LpNormGradient, 
    0 => ("dx", "1D output tensor")
}

args!{LpNormGradient, 
    0 => ("p", "Order of the norm in p-norm"),
    1 => ("average", "whehther we calculate norm or averaged_norm. The Lp_averaged_norm(x) is defined as Lp_averaged_normgradient(x) = LpNormGradient(x) / size(x)")
}

impl<T, Context> LpNormGradientOp<T, Context> {
    
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

