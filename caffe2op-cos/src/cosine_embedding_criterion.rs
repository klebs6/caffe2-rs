crate::ix!();

/**
 | CosineEmbeddingCriterion takes two inputs: the
 | similarity value and the label, and computes the
 | elementwise criterion output as
 |
 | output = 1 - s,               if y == 1
 | max(0, s - margin),  if y == -1
 */
pub struct CosineEmbeddingCriterionOp<Context> {
    storage: OperatorStorage,
    context: Context,
    margin:  f32,
}

num_inputs!{CosineEmbeddingCriterion, 2}

num_outputs!{CosineEmbeddingCriterion, 1}

inputs!{CosineEmbeddingCriterion, 
    0 => ("S", "The cosine similarity as a 1-dim TensorCPU."),
    1 => ("Y", "The label as a 1-dim TensorCPU with int value of 1 or -1.")
}

outputs!{CosineEmbeddingCriterion, 
    0 => ("loss", "The output loss with the same dimensionality as S.")
}
