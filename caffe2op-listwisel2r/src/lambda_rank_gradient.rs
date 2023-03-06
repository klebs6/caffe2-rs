crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LambdaRankNdcgGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{LambdaRankNdcgGradient, 4}

num_outputs!{LambdaRankNdcgGradient, 1}

input_tags!{
    LambdaRankNdcgGradientOp {
        Y,
        SessionLens,
        DyCache,
        Dloss
    }
}

output_tags!{
    LambdaRankNdcgGradientOp {
        Dy
    }
}


