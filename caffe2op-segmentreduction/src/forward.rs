crate::ix!();

/**
  | Incremental reducer ops: assume that
  | reducer consumes pieces of data one
  | by one. Also, supports additional arguments
  | passed to reducer, e.g. scalers for
  | weighted sum.
  | 
  | -----------
  | @note
  | 
  | in current implementation additional
  | inputs are considered auxiliary constants
  | and have limitations:
  | 
  | - there is no gradient computation for
  | auxiliary inputs
  | 
  | - auxiliary inputs aren't affected
  | by fused embedding lookup in operations
  | like sparse_sorted_segment
  |
  */
pub trait HasForwardOp {
    type ForwardOp;
}
