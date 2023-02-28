crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/CopyBytes.h]

pub type CopyBytesFunction = fn(
        nbytes:     usize,
        src:        *const c_void,
        src_device: Device,
        dst:        *mut c_void,
        dst_device: Device
) -> c_void;

pub struct _CopyBytesFunctionRegisterer {

}

macro_rules! register_copy_bytes_function {
    ($from:ident, $to:ident, $($arg:ident),*) => {
        /*
        
          namespace {                                                 
          static _CopyBytesFunctionRegisterer C10_ANONYMOUS_VARIABLE( 
              g_copy_function)(from, to, __VA_ARGS__);                
          }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/CopyBytes.cpp]

/**
  | First dimension of the array is `bool
  | async`: 0 is sync, 1 is async (non-blocking)
  |
  */
lazy_static!{
    /*
    static CopyBytesFunction g_copy_bytes[2][COMPILE_TIME_MAX_DEVICE_TYPES]
                                         [COMPILE_TIME_MAX_DEVICE_TYPES];
    */
}

impl _CopyBytesFunctionRegisterer {
    
    pub fn new(
        from_type:  DeviceType,
        to_type:    DeviceType,
        func_sync:  CopyBytesFunction,
        func_async: CopyBytesFunction) -> Self {
    
        todo!();
        /*


            auto from = static_cast<int>(fromType);
      auto to = static_cast<int>(toType);
      if (!func_async) {
        // default to the sync function
        func_async = func_sync;
      }
      CHECK(
          g_copy_bytes[0][from][to] == nullptr &&
          g_copy_bytes[1][from][to] == nullptr)
          << "Duplicate registration for device type pair "
          << DeviceTypeName(fromType) << ", " << DeviceTypeName(toType);
      g_copy_bytes[0][from][to] = func_sync;
      g_copy_bytes[1][from][to] = func_async;
        */
    }
}

/**
  | WARNING: Implementations for this
  | function are currently registered
  | from
  | 
  | ATen and caffe2, not yet from c10. Don't
  | use this if not either ATen or caffe2
  | is present as well.
  | 
  | We can't move them yet, because the Cuda
  | implementations aren't unified yet
  | between ATen and caffe2.
  | 
  | We're planning to move the implementations
  | into c10/backend/xxx to make c10 self
  | contained again.
  |
  */
pub fn copy_bytes(
    nbytes:     usize,
    src:        *const c_void,
    src_device: Device,
    dst:        *mut c_void,
    dst_device: Device,
    async_:     bool)  {

    todo!();
    /*
            auto ptr = g_copy_bytes[async ? 1 : 0][static_cast<int>(src_device.type())]
                             [static_cast<int>(dst_device.type())];
      CAFFE_ENFORCE(
          ptr,
          "No function found for copying from ",
          DeviceTypeName(src_device.type()),
          " to ",
          DeviceTypeName(dst_device.type()));
      ptr(nbytes, src, src_device, dst, dst_device);
        */
}
