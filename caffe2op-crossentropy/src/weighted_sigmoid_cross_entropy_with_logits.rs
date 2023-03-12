crate::ix!();

/**
  | Given three matrices: logits, targets,
  | weights, all of the same shape, (batch_size,
  | num_classes), computes the weighted
  | sigmoid cross entropy between logits
  | and targets. Specifically, at each
  | position r,c, this computes weights[r,
  | c] * crossentropy(sigmoid(logits[r,
  | c]), targets[r, c]), and then averages
  | over each row.
  | 
  | Returns a tensor of shape (batch_size,)
  | of losses for each example.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSigmoidCrossEntropyWithLogitsOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    phantom: PhantomData<T>,
}

num_inputs!{WeightedSigmoidCrossEntropyWithLogits, 3}

num_outputs!{WeightedSigmoidCrossEntropyWithLogits, 1}

inputs!{WeightedSigmoidCrossEntropyWithLogits, 
    0 => ("logits", "matrix of logits for each example and class."),
    1 => ("targets", "matrix of targets, same shape as logits."),
    2 => ("weights", "matrix of weights, same shape as logits.")
}

outputs!{WeightedSigmoidCrossEntropyWithLogits, 
    0 => ("xentropy", "Vector with the total xentropy for each example.")
}

identical_type_and_shape_of_input_dim!{WeightedSigmoidCrossEntropyWithLogits, (0, 0)}
