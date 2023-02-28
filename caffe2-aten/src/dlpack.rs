crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/dlpack.h]

/**
  | -----------
  | @brief
  | 
  | The current version of dlpack
  |
  */
pub const DLPACK_VERSION: usize = 040;

/**
  | -----------
  | @brief
  | 
  | The device type in DLDevice.
  |
  */
pub enum DLDeviceType {

    /**
      | -----------
      | @brief
      | 
      | CPU device
      |
      */
    kDLCPU = 1,

    /**
      | -----------
      | @brief
      | 
      | CUDA GPU device
      |
      */
    kDLGPU = 2,

    /**
      | -----------
      | @brief
      | 
      | Pinned CUDA GPU device by cudaMallocHost
      | 
      | -----------
      | @note
      | 
      | kDLCPUPinned = kDLCPU | kDLGPU
      |
      */
    kDLCPUPinned = 3,

    /**
      | -----------
      | @brief
      | 
      | OpenCL devices.
      |
      */
    kDLOpenCL = 4,

    /**
      | -----------
      | @brief
      | 
      | Vulkan buffer for next generation graphics.
      |
      */
    kDLVulkan = 7,

    /**
      | -----------
      | @brief
      | 
      | Metal for Apple GPU.
      |
      */
    kDLMetal = 8,

    /**
      | -----------
      | @brief
      | 
      | Verilog simulator buffer
      |
      */
    kDLVPI = 9,

    /**
      | -----------
      | @brief
      | 
      | ROCm GPUs for AMD GPUs
      |
      */
    kDLROCM = 10,

    /**
      | -----------
      | @brief
      | 
      | Reserved extension device type, used
      | for quickly test extension device
      | 
      | The semantics can differ depending
      | on the implementation.
      |
      */
    kDLExtDev = 12,
}

/**
  | -----------
  | @brief
  | 
  | A Device for Tensor and operator.
  |
  */
pub struct DLDevice {

    /**
      | -----------
      | @brief
      | 
      | The device type used in the device.
      |
      */
    device_type: DLDeviceType,

    /**
      | -----------
      | @brief
      | 
      | The device index
      |
      */
    device_id: i32,
}

/**
  | -----------
  | @brief
  | 
  | This is an alias for DLDevice. Notice
  | that this will be removed in the next
  | release.
  |
  */
pub type DLContext = DLDevice;

/**
  | -----------
  | @brief
  | 
  | The type code options DLDataType.
  |
  */
pub enum DLDataTypeCode {

    /**
      | -----------
      | @brief
      | 
      | signed integer
      |
      */
    kDLInt = 0U,

    /**
      | -----------
      | @brief
      | 
      | unsigned integer
      |
      */
    kDLUInt = 1U,

    /**
      | -----------
      | @brief
      | 
      | IEEE floating point
      |
      */
    kDLFloat = 2U,

    /**
      | -----------
      | @brief
      | 
      | Opaque handle type, reserved for testing
      | purposes.
      | 
      | Frameworks need to agree on the handle
      | data type for the exchange to be well-defined.
      |
      */
    kDLOpaqueHandle = 3U,

    /**
      | -----------
      | @brief
      | 
      | bfloat16
      |
      */
    kDLBfloat = 4U,

    /**
      | -----------
      | @brief
      | 
      | complex number (C/C++/Python layout:
      | compact struct per complex number)
      |
      */
    kDLComplex = 5U,
}

/**
  | -----------
  | @brief
  | 
  | The data type the tensor can hold.
  | 
  | Examples
  | 
  | - float: type_code = 2, bits = 32, lanes=1
  | 
  | - float4(vectorized 4 float): type_code
  | = 2, bits = 32, lanes=4
  | 
  | - int8: type_code = 0, bits = 8, lanes=1
  |
  */
pub struct DLDataType {

    /**
      | -----------
      | @brief
      | 
      | Type code of base types.
      | 
      | We keep it u8 instead of DLDataTypeCode
      | for minimal memory footprint, but the
      | value should be one of DLDataTypeCode
      | enum values.
      |
      */
    code:  u8,

    /**
      | -----------
      | @brief
      | 
      | Number of bits, common choices are 8,
      | 16, 32.
      |
      */
    bits:  u8,


    /**
      | -----------
      | @brief
      | 
      | Number of lanes in the type, used for
      | vector types.
      |
      */
    lanes: u16,
}

/**
  | -----------
  | @brief
  | 
  | Plain C Tensor object, does not manage
  | memory.
  |
  */
pub struct DLTensor {

    /**
      | -----------
      | @brief
      | 
      | The opaque data pointer points to the
      | allocated data. This will be
      | 
      | CUDA device pointer or cl_mem handle
      | in OpenCL. This pointer is always aligned
      | to 256 bytes as in CUDA.
      | 
      | For given DLTensor, the size of memory
      | required to store the contents of data
      | is calculated as follows:
      | 
      | 
      | -----------
      | @code
      | 
      | {.c}
      |     static inline usize GetDataSize(const DLTensor* t) {
      |       usize size = 1;
      |       for (tvm_index_t i = 0; i < t->ndim; ++i) {
      |         size *= t->shape[i];
      |       }
      |       size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
      |       return size;
      |     }
      |
      */
    data:        *mut void,

    /**
      | -----------
      | @brief
      | 
      | The device of the tensor
      |
      */
    device:      DLDevice,


    /**
      | -----------
      | @brief
      | 
      | Number of dimensions
      |
      */
    ndim:        i32,


    /**
      | -----------
      | @brief
      | 
      | The data type of the pointer
      |
      */
    dtype:       DLDataType,


    /**
      | -----------
      | @brief
      | 
      | The shape of the tensor
      |
      */
    shape:       *mut i64,


    /**
      | -----------
      | @brief
      | 
      | strides of the tensor (in number of elements,
      | not bytes) can be NULL, indicating tensor
      | is compact and row-majored.
      |
      */
    strides:     *mut i64,


    /**
      | -----------
      | @brief
      | 
      | The offset in bytes to the beginning
      | pointer to data
      |
      */
    byte_offset: u64,

}

/**
  | -----------
  | @brief
  | 
  | C Tensor object, manage memory of DLTensor.
  | 
  | This data structure is intended to facilitate
  | the borrowing of DLTensor by another
  | framework. It is not meant to transfer
  | the tensor.
  | 
  | When the borrowing framework doesn't
  | need the tensor, it should call the deleter
  | to notify the host that the resource
  | is no longer needed.
  |
  */
pub struct DLManagedTensor {

    /**
      | -----------
      | @brief
      | 
      | DLTensor which is being memory managed
      |
      */
    dl_tensor:   DLTensor,

    /**
      | -----------
      | @brief
      | 
      | the context of the original host framework
      | of DLManagedTensor in which DLManagedTensor
      | is used in the framework. It can also
      | be NULL.
      |
      */
    manager_ctx: *mut void,

    /**
      | -----------
      | @brief
      | 
      | Destructor signature void (*)(void*)
      | - this should be called to destruct manager_ctx
      | which holds the DLManagedTensor. It
      | can be NULL if there is no way for the caller
      | to provide a reasonable destructor.
      | 
      | The destructors deletes the argument
      | self as well.
      |
      */
    deleter:     fn(self_: *mut DLManagedTensor) -> void,
}
