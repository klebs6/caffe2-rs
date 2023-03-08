crate::ix!();

/**
  | Map N dim tensor to N+1 dim based on length
  | blob.
  | 
  | Sequences that are shorter than the
  | longest sequence are padded with zeros.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct PackSegmentsOp<Context> {
    storage:                    OperatorStorage,
    context:                    Context,
    max_length:                 i64,
    pad_minf:                   bool,
    padding:                    f32,
    return_presence_mask:       bool,

    /**
      | Scratch space required by the CUDA version
      | 
      | {Context::GetDeviceType()};
      |
      */
    dev_buffer:                 Tensor,

    /// {Context::GetDeviceType()};
    dev_lengths_prefix_sum:     Tensor,

    /// {Context::GetDeviceType()};
    dev_max_length:             Tensor,
    host_max_length:            Tensor, // default = CPU
}
