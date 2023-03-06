crate::ix!();

///----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GroupNormGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    group:    i32,
    order:    StorageOrder,
    ds:       Tensor,
    db:       Tensor,
    dY_scale: Tensor,
    x_scale:  Tensor,
    bias:     Tensor,
    ones:     Tensor,

    /**
      | Input: dY, X, gamma, beta, mu, inv_sig
      | 
      | Output: dX, dgamma, dbeta
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{GroupNormGradient, 6}

num_outputs!{GroupNormGradient, 3}

input_tags!{
    GroupNormGradientOp {
        OutputGrad,
        Input,
        Gamma,
        Beta,
        Mu,
        InvSigma
    }
}

output_tags!{
    GroupNormGradientOp {
        InputGrad,
        GammaGrad,
        BetaGrad
    }
}
