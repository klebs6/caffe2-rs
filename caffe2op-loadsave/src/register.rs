crate::ix!();

register_cpu_operator!{DBExists,   DBExistsOp<CPUContext>}
register_cpu_operator!{Load,       LoadOp<CPUContext>}
register_cpu_operator!{Save,       SaveOp<CPUContext>}
register_cpu_operator!{Checkpoint, CheckpointOp<CPUContext>}

/**
  | CPU Operator old name: do NOT use, we
  | may deprecate this later.
  |
  */
register_cpu_operator!{Snapshot,   CheckpointOp<CPUContext>}

no_gradient!{Load}

should_not_do_gradient!{DBExists}
should_not_do_gradient!{Save}
should_not_do_gradient!{Checkpoint}
should_not_do_gradient!{Snapshot}

register_cuda_operator!{Load,       LoadOp<CUDAContext>}
register_cuda_operator!{Save,       SaveOp<CUDAContext>}
register_cuda_operator!{Checkpoint, CheckpointOp<CUDAContext>}
