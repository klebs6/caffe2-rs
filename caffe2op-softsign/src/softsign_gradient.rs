crate::ix!();

/**
  | Calculates the softsign gradient (sgn(x)/(1+|x|)^2)
  | of the given input tensor element-wise.
  |
  */
pub struct SoftsignGradientFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{SoftsignGradient, 2}

num_outputs!{SoftsignGradient, 1}

inputs!{SoftsignGradient, 
    0 => ("input", "1-D input tensor"),
    1 => ("input", "1-D input tensor")
}

outputs!{SoftsignGradient, 
    0 => ("output", "The softsign gradient (sgn(x)/(1+|x|)^2) 
        values of the input tensor computed element-wise")
}

allow_inplace!{SoftsignGradient, vec![(1, 0)]}
