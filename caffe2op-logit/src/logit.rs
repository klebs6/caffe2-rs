crate::ix!();

/**
  | Elementwise logit transform: logit(x)
  | = log(x / (1 - x)), where x is the input 
  | data clampped in (eps, 1-eps).
  |
  */
pub struct LogitFunctor<Context> {
    eps: f32,

    phantom: PhantomData<Context>,
}

num_inputs!{Logit, 1}

num_outputs!{Logit, 1}

inputs!{Logit, 
    0 => ("X", "input float tensor")
}

outputs!{Logit, 
    0 => ("Y", "output float tensor")
}

args!{Logit, 
    0 => ("eps (optional)", "small positive epsilon value, the default is 1e-6.")
}

identical_type_and_shape!{Logit}

allow_inplace!{Logit, vec![(0, 0)]}

impl<Context> LogitFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : eps_(op.GetSingleArgument<float>("eps", 1e-6f)) 

        CAFFE_ENFORCE_GT(eps_, 0.0);
        CAFFE_ENFORCE_LT(eps_, 0.5);
        */
    }
}

pub type LogitOp = UnaryElementwiseWithArgsOp<
    TensorTypes<f32>,
    CPUContext,
    LogitFunctor<CPUContext>>;

