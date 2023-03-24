crate::ix!();

/**
  | @brief
  | 
  | A struct to store various cost information
  | about an operator such as FLOPs, total
  | memory use and parameters.
  |
  */
pub struct OpSchemaCost {

    /// Floating point operations.
    flops: u64,

    /// Total memory read.
    bytes_read: u64,

    /// Total memory written.
    bytes_written: u64,

    /// Memory read for parameters.
    params_bytes: u64,
}
