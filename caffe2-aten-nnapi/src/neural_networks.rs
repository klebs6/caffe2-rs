crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/nnapi/NeuralNetworks.h]

/**
  | Most of NeuralNetworks.h has been stripped
  | for simplicity.
  | 
  | We don't need any of the function declarations
  | since we call them all through dlopen/dlsym.
  | 
  | Operation codes are pulled directly
  | from serialized models.
  |
  */
pub enum ResultCode {
    ANEURALNETWORKS_NO_ERROR                 = 0,
    ANEURALNETWORKS_OUT_OF_MEMORY            = 1,
    ANEURALNETWORKS_INCOMPLETE               = 2,
    ANEURALNETWORKS_UNEXPECTED_NULL          = 3,
    ANEURALNETWORKS_BAD_DATA                 = 4,
    ANEURALNETWORKS_OP_FAILED                = 5,
    ANEURALNETWORKS_BAD_STATE                = 6,
    ANEURALNETWORKS_UNMAPPABLE               = 7,
    ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE = 8,
    ANEURALNETWORKS_UNAVAILABLE_DEVICE       = 9,
}

pub enum OperandCode {
    ANEURALNETWORKS_FLOAT32                        = 0,
    ANEURALNETWORKS_INT32                          = 1,
    ANEURALNETWORKS_UINT32                         = 2,
    ANEURALNETWORKS_TENSOR_FLOAT32                 = 3,
    ANEURALNETWORKS_TENSOR_INT32                   = 4,
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM            = 5,
    ANEURALNETWORKS_BOOL                           = 6,
    ANEURALNETWORKS_TENSOR_QUANT16_SYMM            = 7,
    ANEURALNETWORKS_TENSOR_FLOAT16                 = 8,
    ANEURALNETWORKS_TENSOR_BOOL8                   = 9,
    ANEURALNETWORKS_FLOAT16                        = 10,
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
    ANEURALNETWORKS_TENSOR_QUANT16_ASYMM           = 12,
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM             = 13,
}

pub enum PreferenceCode {
    ANEURALNETWORKS_PREFER_LOW_POWER          = 0,
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED    = 2,
}

pub type ANeuralNetworksMemory        = ANeuralNetworksMemory;
pub type ANeuralNetworksModel         = ANeuralNetworksModel;
pub type ANeuralNetworksDevice        = ANeuralNetworksDevice;
pub type ANeuralNetworksCompilation   = ANeuralNetworksCompilation;
pub type ANeuralNetworksExecution     = ANeuralNetworksExecution;
pub type ANeuralNetworksEvent         = ANeuralNetworksEvent;
pub type ANeuralNetworksOperationType = i32;

pub struct ANeuralNetworksOperandType {
    ty:              i32,
    dimension_count: u32,
    dimensions:      *const u32,
    scale:           f32,
    zero_point:      i32,
}
