crate::ix!();

/**
  | Group Normalization (GN) operation:
  | https://arxiv.org/abs/1803.08494
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GroupNormOp<T, Context> {

    storage:  OperatorStorage,
    context:  Context,

    group:    i32,
    epsilon:  f32,
    order:    StorageOrder,
    is_test:  bool,
    mu:       Tensor,
    rsig:     Tensor,
    scale:    Tensor,
    bias:     Tensor,

    /**
      | Input: X, gamma, beta
      | 
      | Output: Y, mu, inv_sig
      |
      */
    phantom:  PhantomData<T>,
}

/**
  | Input: X, gamma, beta;
  | 
  | Output: Y, mu, sig
  |
  */
num_inputs!{GroupNorm, 3}

num_outputs!{GroupNorm, (1,3)}

inputs!{GroupNorm, 
    0 => ("X",      ">=4D feature map input of shape (N, C, H, W) or (N, C, T, H, W)"),
    1 => ("gamma",  "The scale as a 1-dimensional tensor of size C to be applied to the output."),
    2 => ("beta",   "The bias as a 1-dimensional tensor of size C to be applied to the output.")
}

outputs!{GroupNorm, 
    0 => ("Y",      "The output >=4-dimensional tensor of the same shape as X."),
    1 => ("mean",   "The mean of shape (N, G). For backward usage or reference. Cannot be used as activations."),
    2 => ("std",    "The std of shape (N, G). For backward usage or reference. Cannot be used as activations.")
}

args!{GroupNorm, 
    0 => ("num_groups", "(int) default 32; number of groups used by GN."),
    1 => ("epsilon",    "(float) default 1e-5; small constant added to var.")
}

input_tags!{
    GroupNormOp {
        Input,
        Gamma,
        Beta
    }
}

output_tags!{
    GroupNormOp {
        Output,
        Mu,
        InvSigma
    }
}

register_cpu_operator!{
    GroupNorm, 
    GroupNormOp<f32, CPUContext>
}

register_cpu_operator!{
    GroupNormGradient,
    GroupNormGradientOp<f32, CPUContext>
}
