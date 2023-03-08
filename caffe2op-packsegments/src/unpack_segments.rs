crate::ix!();

/**
  | Map N+1 dim tensor to N dim based on length
  | blob
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct UnpackSegmentsOp<Context> {
    storage:                 OperatorStorage,
    context:                 Context,
    max_length:              i64,
    dev_buffer:              Tensor, // {Context::GetDeviceType()};
    dev_lengths_prefix_sum:  Tensor, // {Context::GetDeviceType()};
    dev_max_length:          Tensor, // {Context::GetDeviceType()};
    dev_num_cell:            Tensor, // {Context::GetDeviceType()};
    host_max_length:         Tensor, // {CPU};
    host_num_cell:           Tensor, // {CPU};
}
