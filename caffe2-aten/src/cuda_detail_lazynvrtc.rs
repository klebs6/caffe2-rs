crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/detail/LazyNVRTC.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/detail/LazyNVRTC.cpp]

pub fn get_cuda_library() -> &mut DynamicLibrary {
    
    todo!();
        /*
            #if defined(_WIN32)
      static at::DynamicLibrary lib("nvcuda.dll");
    #else
      static at::DynamicLibrary lib("libcuda.so.1");
    #endif
      return lib;
        */
}

pub fn get_lib_version() -> String {
    
    todo!();
        /*
            // [NVRTC versioning]
      // Quote of https://docs.nvidia.com/cuda/nvrtc/index.html Section 8.1. NVRTC library versioning
      //
      // In the following, MAJOR and MINOR denote the major and minor versions of the CUDA Toolkit.
      // e.g. for CUDA 11.2, MAJOR is "11" and MINOR is "2".
      //
      // Linux:
      //   - In CUDA toolkits prior to CUDA 11.3, the soname was set to "MAJOR.MINOR".
      //   - In CUDA 11.3 and later 11.x toolkits, the soname field is set to "11.2".
      //   - In CUDA toolkits with major version > 11 (e.g. CUDA 12.x), the soname field is set to "MAJOR".
      //
      // Windows:
      //   - In CUDA toolkits prior to cuda 11.3, the DLL name was of the form "nvrtc64_XY_0.dll", where X = MAJOR, Y = MINOR.
      //   - In CUDA 11.3 and later 11.x toolkits, the DLL name is "nvrtc64_112_0.dll".
      //   - In CUDA toolkits with major version > 11 (e.g. CUDA 12.x), the DLL name is of the form "nvrtc64_X0_0.dll" where X = MAJOR.
      //
      // Consider a CUDA toolkit with major version > 11. The NVRTC library in this CUDA toolkit will have the same soname (Linux)
      // or DLL name (Windows) as an NVRTC library in a previous minor version of the same CUDA toolkit. Similarly, the NVRTC
      // library in CUDA 11.3 and later 11.x releases will have the same soname (Linux) or DLL name (Windows) as the NVRTC library in CUDA 11.2.
      constexpr auto major = CUDA_VERSION / 1000;
      constexpr auto minor = ( CUDA_VERSION / 10 ) % 10;
    #if defined(_WIN32)
      if (major < 11 || (major == 11 && minor < 3)) {
        return std::to_string(major) + std::to_string(minor);
      } else if (major == 11) {
        return "112";
      } else {
        return std::to_string(major) + "0";
      }
    #else
      if (major < 11 || (major == 11 && minor < 3)) {
        return std::to_string(major) + "." + std::to_string(minor);
      } else if (major == 11) {
        return "11.2";
      } else {
        return std::to_string(major);
      }
    #endif
        */
}

pub fn get_lib_name() -> String {
    
    todo!();
        /*
            #if defined(_WIN32)
      return std::string("nvrtc64_") + getLibVersion() + "_0.dll";
    #else
      return std::string("libnvrtc.so.") + getLibVersion();
    #endif
        */
}

pub fn get_alt_lib_name() -> String {
    
    todo!();
        /*
            #if !defined(_WIN32) && defined(NVRTC_SHORTHASH)
      return std::string("libnvrtc-") + C10_STRINGIZE(NVRTC_SHORTHASH) + ".so." + getLibVersion();
    #else
      return {};
    #endif
        */
}

pub fn get_nvrtc_library() -> &mut DynamicLibrary {
    
    todo!();
        /*
            static std::string libname = getLibName();
      static std::string alt_libname = getAltLibName();
      static at::DynamicLibrary lib(libname.c_str(), alt_libname.empty() ? nullptr : alt_libname.c_str());
      return lib;
        */
}

#[macro_export] macro_rules! _stub_1 {
    ($LIB:ty, $NAME:ty, $RETTYPE:ty, $ARG1:ty) => {
        /*
        
        RETTYPE NAME(ARG1 a1) {                                                              
          auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); 
          if (!fn)                                                                           
            throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     
          lazyNVRTC.NAME = fn;                                                               
          return fn(a1);                                                                     
        }
        */
    }
}

#[macro_export] macro_rules! _stub_2 {
    ($LIB:ty, $NAME:ty, $RETTYPE:ty, $ARG1:ty, $ARG2:ty) => {
        /*
        
        RETTYPE NAME(ARG1 a1, ARG2 a2) {                                                     
          auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); 
          if (!fn)                                                                           
            throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     
          lazyNVRTC.NAME = fn;                                                               
          return fn(a1, a2);                                                                 
        }
        */
    }
}

#[macro_export] macro_rules! _stub_3 {
    ($LIB:ty, $NAME:ty, $RETTYPE:ty, $ARG1:ty, $ARG2:ty, $ARG3:ty) => {
        /*
        
        RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3) {                                            
          auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); 
          if (!fn)                                                                           
            throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     
          lazyNVRTC.NAME = fn;                                                               
          return fn(a1, a2, a3);                                                             
        }
        */
    }
}


#[macro_export] macro_rules! _stub_4 {
    ($LIB:ty, $NAME:ty, $RETTYPE:ty, $ARG1:ty, $ARG2:ty, $ARG3:ty, $ARG4:ty) => {
        /*
        
        RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4) {                                   
          auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); 
          if (!fn)                                                                           
            throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     
          lazyNVRTC.NAME = fn;                                                               
          return fn(a1, a2, a3, a4);                                                         
        }
        */
    }
}

#[macro_export] macro_rules! cuda_stub1 {
    ($NAME:ty, $A1:ty) => {
        /*
                _STUB_1(CUDA, NAME, CUresult CUDAAPI, A1)
        */
    }
}

#[macro_export] macro_rules! cuda_stub2 {
    ($NAME:ty, $A1:ty, $A2:ty) => {
        /*
                _STUB_2(CUDA, NAME, CUresult CUDAAPI, A1, A2)
        */
    }
}

#[macro_export] macro_rules! cuda_stub3 {
    ($NAME:ty, $A1:ty, $A2:ty, $A3:ty) => {
        /*
                _STUB_3(CUDA, NAME, CUresult CUDAAPI, A1, A2, A3)
        */
    }
}

#[macro_export] macro_rules! cuda_stub4 {
    ($NAME:ty, $A1:ty, $A2:ty, $A3:ty, $A4:ty) => {
        /*
                _STUB_4(CUDA, NAME, CUresult CUDAAPI, A1, A2, A3, A4)
        */
    }
}

#[macro_export] macro_rules! nvrtc_stub1 {
    ($NAME:ty, $A1:ty) => {
        /*
                _STUB_1(NVRTC, NAME, nvrtcResult, A1)
        */
    }
}

#[macro_export] macro_rules! nvrtc_stub2 {
    ($NAME:ty, $A1:ty, $A2:ty) => {
        /*
                _STUB_2(NVRTC, NAME, nvrtcResult, A1, A2)
        */
    }
}

#[macro_export] macro_rules! nvrtc_stub3 {
    ($NAME:ty, $A1:ty, $A2:ty, $A3:ty) => {
        /*
                _STUB_3(NVRTC, NAME, nvrtcResult, A1, A2, A3)
        */
    }
}

nvrtc_stub2!(nvrtcVersion, *mut i32, *mut i32);

nvrtc_stub2!(nvrtcAddNameExpression, nvrtcProgram, *const u8);

pub fn nvrtc_create_program(
        prog:          *mut NvrtcProgram,
        src:           *const u8,
        name:          *const u8,
        num_headers:   i32,
        headers:       *const *const u8,
        include_names: *const *const u8) -> NvrtcResult {
    
    todo!();
        /*
            auto fn = reinterpret_cast<decltype(&nvrtcCreateProgram)>(getNVRTCLibrary().sym(__func__));
      if (!fn)
        throw std::runtime_error("Can't get nvrtcCreateProgram");
      lazyNVRTC.nvrtcCreateProgram = fn;
      return fn(prog, src, name, numHeaders, headers, includeNames);
        */
}

nvrtc_stub1!(
    nvrtcDestroyProgram, 
    *mut nvrtcProgram
);

nvrtc_stub2!(
    nvrtcGetPTXSize, 
    nvrtcProgram, 
    *mut usize
);

nvrtc_stub2!(
    nvrtcGetPTX, 
    nvrtcProgram, 
    *mut u8
);

#[cfg(CUDA_VERSION_gte_11010)]
nvrtc_stub2!(nvrtcGetCUBINSize, nvrtcProgram, *mut usize);

#[cfg(CUDA_VERSION_gte_11010)]
nvrtc_stub2!(nvrtcGetCUBIN, nvrtcProgram, *mut u8);

nvrtc_stub3!{
    nvrtcCompileProgram, 
    nvrtcProgram, 
    i32, 
    *const *const u8
}

_stub_1!{
    NVRTC, 
    nvrtcGetErrorString, 
    *const u8, 
    nvrtcResult
}

nvrtc_stub2!{
    nvrtcGetProgramLogSize,
    nvrtcProgram, 
    *mut usize
}

nvrtc_stub2!{
    nvrtcGetProgramLog, 
    nvrtcProgram, 
    *mut u8
}

nvrtc_stub3!{
    nvrtcGetLoweredName, 
    nvrtcProgram, 
    *const u8, 
    *const *const u8
}

cuda_stub2!{
    cuModuleLoadData, 
    *mut CUmodule, 
    *const c_void
}

cuda_stub3!{
    cuModuleGetFunction, 
    *mut CUfunction, 
    CUmodule, 
    *const u8
}

cuda_stub4!{
    cuOccupancyMaxActiveBlocksPerMultiprocessor, 
    *mut i32, 
    CUfunction, 
    i32, 
    usize
}

cuda_stub2!{
    cuGetErrorString, 
    CUresult, 
    *const *const u8
}

cuda_stub1!{
    cuCtxGetCurrent, 
    *mut CUcontext
}

cuda_stub1!{
    cuModuleUnload, 
    CUmodule
}

cuda_stub3!{
    cuDevicePrimaryCtxGetState, 
    CUdevice, 
    *mut u32, 
    *mut i32
}

cuda_stub4!{
    cuLinkCreate, 
    u32, 
    *mut CUjit_option, 
    *mut *mut c_void, 
    *mut CUlinkState
}

cuda_stub3!{
    cuLinkComplete, 
    CUlinkState, 
    *mut *mut c_void, 
    *mut usize
}

/// Irregularly shaped functions
///
pub fn cu_launch_kernel(
    f:                CUfunction,
    grid_dimx:        u32,
    grid_dimy:        u32,
    grid_dimz:        u32,
    block_dimx:       u32,
    block_dimy:       u32,
    block_dimz:       u32,
    shared_mem_bytes: u32,
    h_stream:         CUstream,
    kernel_params:    *mut *mut void,
    extra:            *mut *mut void) -> CUresult {
    
    todo!();
        /*
            auto fn = reinterpret_cast<decltype(&cuLaunchKernel)>(getCUDALibrary().sym(__func__));
      if (!fn)
        throw std::runtime_error("Can't get cuLaunchKernel");
      lazyNVRTC.cuLaunchKernel = fn;
      return fn(f,
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);
        */
}


pub fn cu_module_load_data_ex(
        module:        *mut CUmodule,
        image:         *const void,
        num_options:   u32,
        options:       *mut CUjit_option,
        option_values: *mut *mut void) -> CUresult {
    
    todo!();
        /*
            auto fn = reinterpret_cast<decltype(&cuModuleLoadDataEx)>(getCUDALibrary().sym(__func__));
      if (!fn)
        throw std::runtime_error("Can't get cuModuleLoadDataEx");
      lazyNVRTC.cuModuleLoadDataEx = fn;
      return fn(module, image, numOptions, options, optionValues);
        */
}

pub fn cu_link_add_data(
        state:         CUlinkState,
        ty:            CUjitInputType,
        data:          *mut void,
        size:          usize,
        name:          *const u8,
        num_options:   u32,
        options:       *mut CUjit_option,
        option_values: *mut *mut void) -> CUresult {
    
    todo!();
        /*
            auto fn = reinterpret_cast<decltype(&cuLinkAddData)>(getCUDALibrary().sym(__func__));
      if (!fn)
        throw std::runtime_error("Can't get cuLinkAddData");
      lazyNVRTC.cuLinkAddData = fn;
      return fn(state, type, data, size, name, numOptions, options, optionValues);
        */
}

lazy_static!{
    /*
    NVRTC lazyNVRTC = {
    #define _REFERENCE_MEMBER(name) _stubs::name,
      AT_FORALL_NVRTC(_REFERENCE_MEMBER)
    #undef _REFERENCE_MEMBER
    };
    */
}
