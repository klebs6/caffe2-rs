crate::ix!();

register_cpu_operator!{Partition,        PartitionOp}
register_cpu_operator!{LengthsPartition, LengthsPartitionOp}
register_cpu_operator!{GatherByKey,      GatherByKeyOp}

/**
  | This should actually have gradient, but for
  | now nothing uses it.
  |
  | Because gradient computation right now is not
  | input/output aware it can't be
  | GRADIENT_NOT_IMPLEMENTEDYET
  */
no_gradient!{Partition}
no_gradient!{LengthsPartition}
register_gradient!{GatherByKey, GetGatherByKeyGradient}
