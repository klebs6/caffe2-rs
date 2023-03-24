crate::ix!();

/**
  | The operator registry. Since we are
  | not expecting a great number of devices,
  | we will simply have an if-then type command
  | and allocate the actual generation
  | to device-specific registerers.
  | 
  | -----------
  | @note
  | 
  | although we have CUDA and CUDNN here,
  | the registerers themselves do not depend
  | on specific cuda or cudnn libraries.
  | This means that we will be able to compile
  | it even when there is no cuda available
  | - we simply do not link any cuda or cudnn
  | operators.
  |
  */
declare_registry!{
    CPUOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

declare_registry!{
    CUDAOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

// Macros for HIP operators
declare_registry!{
    HIPOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}
