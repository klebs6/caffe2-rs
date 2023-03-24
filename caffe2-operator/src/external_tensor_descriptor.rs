crate::ix!();

/**
  | This is for transferring tensor data
  | between C2 and backends.
  |
  */
#[cfg(not(c10_mobile))]
pub struct ExternalTensorDescriptor {
    data_type:            u64,
    dimensions:           u32,
    shape:                *const u64,
    is_offline:           u8, // default = 0
    quantization_axis:    u32,
    quantization_params:  u64,
    scales:               *const f32,
    biases:               *const i32,
    buffer:               u64,
}

