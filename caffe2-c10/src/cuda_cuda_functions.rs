/*!
  | This header provides C++ wrappers around
  | commonly used Cuda API functions.
  |
  | The benefit of using C++ here is that we can
  | raise an exception in the event of an error,
  | rather than explicitly pass around error codes.
  | This leads to more natural APIs.
  |
  | The naming convention used here matches the
  | naming convention of torch.cuda
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDAFunctions.h]
//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDAFunctions.cpp]

/// returns -1 on failure
pub fn driver_version() -> i32 {
    
    todo!();
        /*
            int driver_version = -1;
      cudaError_t err = cudaDriverGetVersion(&driver_version);
      if (err != cudaSuccess) {
        cudaError_t last_err  = cudaGetLastError();
      }
      return driver_version;
        */
}

pub fn device_count_impl(fail_if_no_driver: bool) -> i32 {
    
    todo!();
        /*
            int count;
      auto err = cudaGetDeviceCount(&count);
      if (err == cudaSuccess) {
        return count;
      }
      // Clear out the error state, so we don't spuriously trigger someone else.
      // (This shouldn't really matter, since we won't be running very much Cuda
      // code in this regime.)
      cudaError_t last_err  = cudaGetLastError();
      switch (err) {
        case cudaErrorNoDevice:
          // Zero devices is ok here
          count = 0;
          break;
        case cudaErrorInsufficientDriver: {
          auto version = driver_version();
          if (version <= 0) {
            if (!fail_if_no_driver) {
              // No Cuda driver means no devices
              count = 0;
              break;
            }
            TORCH_CHECK(
                false,
                "Found no NVIDIA driver on your system. Please check that you "
                "have an NVIDIA GPU and installed a driver from "
                "http://www.nvidia.com/Download/index.aspx");
          } else {
            TORCH_CHECK(
                false,
                "The NVIDIA driver on your system is too old (found version ",
                version,
                "). Please update your GPU driver by downloading and installing "
                "a new version from the URL: "
                "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
                "https://pytorch.org to install a PyTorch version that has been "
                "compiled with your version of the Cuda driver.");
          }
        } break;
        case cudaErrorInitializationError:
          TORCH_CHECK(
              false,
              "Cuda driver initialization failed, you might not "
              "have a Cuda gpu.");
          break;
        case cudaErrorUnknown:
          TORCH_CHECK(
              false,
              "Cuda unknown error - this may be due to an "
              "incorrectly set up environment, e.g. changing env "
              "variable CUDA_VISIBLE_DEVICES after program start. "
              "Setting the available devices to be zero.");
          break;
    #if C10_ASAN_ENABLED
        case cudaErrorMemoryAllocation:
          // In ASAN mode, we know that a cudaErrorMemoryAllocation error will
          // pop up if compiled with NVCC (clang-cuda is fine)
          TORCH_CHECK(
              false,
              "Got 'out of memory' error while trying to initialize Cuda. "
              "Cuda with nvcc does not work well with ASAN and it's probably "
              "the reason. We will simply shut down Cuda support. If you "
              "would like to use GPUs, turn off ASAN.");
          break;
    #endif // C10_ASAN_ENABLED
        default:
          TORCH_CHECK(
              false,
              "Unexpected error from cudaGetDeviceCount(). Did you run "
              "some cuda functions before calling NumCudaDevices() "
              "that might have already set an error? Error ",
              err,
              ": ",
              cudaGetErrorString(err));
      }
      return count;
        */
}

/**
  | NB: In the past, we were inconsistent about
  | whether or not this reported an error if there
  | were driver problems are not.  Based on
  | experience interacting with users, it seems
  | that people basically ~never want this function
  | to fail; it should just return zero if things
  | are not working.
  |
  | Oblige them.
  |
  | It still might log a warning for user first
  | time it's invoked
  */
pub fn device_count() -> DeviceIndex {
    
    todo!();
        /*
            // initialize number of devices only once
      static int count = []() {
        try {
          auto result = device_count_impl(/*fail_if_no_driver=*/false);
          TORCH_INTERNAL_ASSERT(
              result <= DeviceIndex::max,
              "Too many Cuda devices, DeviceIndex overflowed");
          return result;
        } catch (const Error& ex) {
          // We don't want to fail, but still log the warning
          // msg() returns the message without the stack trace
          TORCH_WARN("Cuda initialization: ", ex.msg());
          return 0;
        }
      }();
      return static_cast<DeviceIndex>(count);
        */
}

/**
  | Version of device_count that throws
  | is no devices are detected
  |
  */
pub fn device_count_ensure_non_zero() -> DeviceIndex {
    
    todo!();
        /*
            // Call the implementation every time to throw the exception
      int count = device_count_impl(/*fail_if_no_driver=*/true);
      // Zero gpus doesn't produce a warning in `device_count` but we fail here
      TORCH_CHECK(count, "No Cuda GPUs are available");
      return static_cast<DeviceIndex>(count);
        */
}

pub fn current_device() -> DeviceIndex {
    
    todo!();
        /*
            int cur_device;
      C10_CUDA_CHECK(cudaGetDevice(&cur_device));
      return static_cast<DeviceIndex>(cur_device);
        */
}

pub fn set_device(device: DeviceIndex)  {
    
    todo!();
        /*
            C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(device)));
        */
}

pub fn device_synchronize()  {
    
    todo!();
        /*
            C10_CUDA_CHECK(cudaDeviceSynchronize());
        */
}

pub fn get_cuda_check_suffix() -> *const u8 {
    
    todo!();
        /*
            static char* device_blocking_flag = getenv("CUDA_LAUNCH_BLOCKING");
      static bool blocking_enabled =
          (device_blocking_flag && atoi(device_blocking_flag));
      if (blocking_enabled) {
        return "";
      } else {
        return "\nCUDA kernel errors might be asynchronously reported at some"
               " other API call,so the stacktrace below might be incorrect."
               "\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1.";
      }
        */
}
