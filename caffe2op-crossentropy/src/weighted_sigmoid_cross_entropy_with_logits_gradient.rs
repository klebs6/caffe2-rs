crate::ix!();

///------------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSigmoidCrossEntropyWithLogitsGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    phantom: PhantomData<T>,
}

num_inputs!{WeightedSigmoidCrossEntropyWithLogitsGradient, 4}

num_outputs!{WeightedSigmoidCrossEntropyWithLogitsGradient, 1}
