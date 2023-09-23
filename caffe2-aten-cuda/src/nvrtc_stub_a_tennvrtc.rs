/*!
  | NOTE [ USE OF NVRTC AND DRIVER API ]
  |
  | ATen does not directly link to either libnvrtc
  | or libcuda because they require libcuda to be
  | installed, yet we want our GPU build to work on
  | CPU machines as long as CUDA is not
  | initialized.
  |
  | Normal CUDA code in torch uses the cuda runtime
  | libraries which can be installed even if the
  | driver is not installed, but sometimes we
  | specifically need to use the driver API (e.g.,
  | to load JIT compiled code).
  |
  | To accomplish this, we lazily link
  | libcaffe2_nvrtc which provides a struct
  | at::cuda::NVRTC that contains function pointers
  | to all of the apis we need.
  |
  | IT IS AN ERROR TO TRY TO CALL ANY nvrtc* or cu* FUNCTION DIRECTLY.
  | INSTEAD USE, e.g.
  |   detail::getCUDAHooks().nvrtc().cuLoadModule(...)
  | or
  |   globalContext().getNVRTC().cuLoadModule(...)
  |
  | If a function is missing add it to the list in
  | ATen/cuda/nvrtc_stub/ATenNVRTC.h and edit
  | ATen/cuda/detail/LazyNVRTC.cpp accordingly
  | (e.g., via one of the stub macros).
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.h]

#[cfg(not(__HIP_PLATFORM_HCC__))]
#[macro_export] macro_rules! at_forall_nvrtc_base {
    ($_:ident) => {
        /*
        
          _(nvrtcVersion)                                
          _(nvrtcAddNameExpression)                      
          _(nvrtcCreateProgram)                          
          _(nvrtcDestroyProgram)                         
          _(nvrtcGetPTXSize)                             
          _(nvrtcGetPTX)                                 
          _(nvrtcCompileProgram)                         
          _(nvrtcGetErrorString)                         
          _(nvrtcGetProgramLogSize)                      
          _(nvrtcGetProgramLog)                          
          _(nvrtcGetLoweredName)                         
          _(cuModuleLoadData)                            
          _(cuModuleLoadDataEx)                          
          _(cuModuleGetFunction)                         
          _(cuOccupancyMaxActiveBlocksPerMultiprocessor) 
          _(cuGetErrorString)                            
          _(cuLaunchKernel)                              
          _(cuCtxGetCurrent)                             
          _(cuModuleUnload)                              
          _(cuDevicePrimaryCtxGetState)                  
          _(cuLinkCreate)                                
          _(cuLinkAddData)                               
          _(cuLinkComplete)
        */
    }
}

#[cfg(not(__HIP_PLATFORM_HCC__))]
#[cfg(CUDA_VERSION_gte_11010)]
#[macro_export] macro_rules! at_forall_nvrtc {
    ($_:ident) => {
        /*
        
          AT_FORALL_NVRTC_BASE(_)  
          _(nvrtcGetCUBINSize)     
          _(nvrtcGetCUBIN)
        */
    }
}

#[cfg(not(__HIP_PLATFORM_HCC__))]
#[cfg(not(CUDA_VERSION_gte_11010))]
#[macro_export] macro_rules! at_forall_nvrtc {
    ($_:ident) => {
        /*
        
          AT_FORALL_NVRTC_BASE(_)
        */
    }
}

// NOTE [ ATen NVRTC Stub and HIP ]
//
// ATen's NVRTC stub library, caffe2_nvrtc,
// provides dynamic loading of both NVRTC and
// driver APIs. While the former is not yet
// supported for HIP, the later is supported and
// needed (e.g., in
// CUDAHooks::getDeviceWithPrimaryContext() used
// by tensor.pin_memory()).
//
// The macro below strips out certain unsupported
// operations on HIP from the full list above.
//
// HIP doesn't have
//   cuGetErrorString  (maps to non-functional hipGetErrorString___)
//
// HIP from ROCm 3.5 on renamed
// hipOccupancyMaxActiveBlocksPerMultiprocessor to
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
//
#[cfg(__HIP_PLATFORM_HCC__)]
#[cfg(HIP_VERSION_lt_305)]
#[macro_export] macro_rules! hipoccupancymaxactiveblockspermultiprocessor {
    () => {
        /*
                hipOccupancyMaxActiveBlocksPerMultiprocessor
        */
    }
}

#[cfg(__HIP_PLATFORM_HCC__)]
#[cfg(not(HIP_VERSION_lt_305))]
#[macro_export] macro_rules! hipoccupancymaxactiveblockspermultiprocessor {
    () => {
        /*
                cuOccupancyMaxActiveBlocksPerMultiprocessor
        */
    }
}

#[cfg(__HIP_PLATFORM_HCC__)]
#[macro_export] macro_rules! at_forall_nvrtc {
    ($_:ident) => {
        /*
        
          _(nvrtcVersion)                                 
          _(nvrtcCreateProgram)                           
          _(nvrtcAddNameExpression)                       
          _(nvrtcDestroyProgram)                          
          _(nvrtcGetPTXSize)                              
          _(nvrtcGetPTX)                                  
          _(cuModuleLoadData)                             
          _(cuModuleGetFunction)                          
          _(HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR) 
          _(nvrtcGetErrorString)                          
          _(nvrtcGetProgramLogSize)                       
          _(nvrtcGetProgramLog)                           
          _(cuLaunchKernel)                               
          _(nvrtcCompileProgram)                          
          _(cuCtxGetCurrent)                              
          _(nvrtcGetLoweredName)                          
          _(cuModuleUnload)                               
          _(cuDevicePrimaryCtxGetState)
        */
    }
}

lazy_static!{
    /*
    extern "C" typedef struct NVRTC {
    #define CREATE_MEMBER(name) decltype(&name) name;
      AT_FORALL_NVRTC(CREATE_MEMBER)

    } NVRTC;

    extern "C" TORCH_CUDA_CPP_API NVRTC* load_nvrtc();
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp]

pub fn load_nvrtc() -> *mut NVRTC {
    
    todo!();
        /*
            auto self = new NVRTC();
    #define CREATE_ASSIGN(name) self->name = name;
      AT_FORALL_NVRTC(CREATE_ASSIGN)
      return self;
        */
}
