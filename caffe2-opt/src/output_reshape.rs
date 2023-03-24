crate::ix!();

/**
  | Provides slicing info for the outputs.
  | All the vector members should be of the
  | same size as number of outputs of the
  | Onnxifi op.
  |
  */
pub struct OutputReshapeInfo {
    begins:     Vec<Tensor>,
    ends:       Vec<Tensor>,
    fast_path:  Vec<bool>,
}
