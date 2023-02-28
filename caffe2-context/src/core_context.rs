crate::ix!();

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
pub type RandGenType = CPUGeneratorImpl;

#[cfg(any(caffe2_is_xplat_build, c10_mobile))]
pub type RandGenType = mersenne_twister::MT19937;

/**
  | The CPU Context, representing the bare
  | minimum of what a Context class in Caffe2
  | should implement. 
  |
  | // TODO modify docs
  | 
  | See operator.h, especially Operator<Context>,
  | for how Context are used in actual operator
  | implementations that are associated
  | with specific devices.
  | 
  | In general, the Context class is passed
  | in as a template argument, and the operator
  | can use the functions defined in the
  | context to execute whatever computation
  | it has.
  |
  */
#[derive(Default)]
pub struct CPUContext {

    /**
      | TODO(jiayq): instead of hard-coding
      | a generator, make it more flexible.
      |
      */
    random_seed:      i32, //1701
    random_seed_set:  bool,//false
    random_generator: Box<RandGenType>,
}

impl From<&DeviceOption> for CPUContext {
    fn from(x: &DeviceOption) -> CPUContext {
        todo!();
        /*
            : random_seed_(option.has_random_seed() ? option.random_seed() : 1701),
            random_seed_set_(option.has_random_seed() ? true : false) 

        CAFFE_ENFORCE_EQ(option.device_type(), PROTO_CPU);
        */
    }
}

impl From<&Device> for CPUContext {
    fn from(x: &Device) -> CPUContext {
        todo!();
        /*
            : CPUContext(DeviceToOption(device))
        */
    }
}

impl CPUContext {
    
    #[inline] pub fn supports_non_fundamental_types(&self) -> bool {
        
        todo!();
        /*
            // CPU non fumdamental type copy OK
        return true;
        */
    }
    
    #[inline] pub fn device(&self) -> Device {
        
        todo!();
        /*
            // TODO: numa?
        return at::Device(CPU);
        */
    }
    
    #[inline] pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return CPU;
        */
    }
    
    #[inline] pub fn get_device_type() -> DeviceType {
        
        todo!();
        /*
            return CPU;
        */
    }
    
    #[inline] pub fn new(nbytes: usize) -> DataPtr {
        
        todo!();
        /*
            return GetCPUAllocator()->allocate(nbytes);
        */
    }
    
    #[inline] pub fn copy_bytes_fromCPU(
        &mut self, 
        nbytes: usize,
        src:    *const c_void,
        dst:    *mut c_void)  
    {
        todo!();
        /*
            CopyBytesSameDevice(nbytes, src, dst);
        */
    }
    
    #[inline] pub fn copy_bytes_toCPU(
        &mut self, 
        nbytes: usize,
        src:    *const c_void,
        dst:    *mut c_void)  
    {
        todo!();
        /*
            CopyBytesSameDevice(nbytes, src, dst);
        */
    }

    #[inline] pub fn copy<T, SrcContext, DstContext>(
        &mut self, 
        n:   usize,
        src: *const T,
        dst: *mut T) 
    {
        todo!();
        /*
            if (c10::guts::is_fundamental<T>::value) {
              CopyBytes<SrcContext, DstContext>(
                  n * sizeof(T),
                  static_cast<const void*>(src),
                  static_cast<void*>(dst));
            } else {
              for (size_t i = 0; i < n; ++i) {
                dst[i] = src[i];
              }
            }
        */
    }

    #[inline] pub fn copy_items<SrcContext, DstContext>(
        &mut self,
        meta: TypeMeta,
        n:    usize,
        src:  *const c_void,
        dst:  *mut c_void) {
        todo!();
        /*
            if (meta.copy()) {
              meta.copy()(src, dst, n);
            } else {
              CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
            }
        */
    }
    
    /// By default CPU operators don't have async device parts
    #[inline] pub fn has_async_part_default() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] pub fn supports_async_scheduling() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    /**
      | CPU streams are not implemented and
      | are silently ignored by CPU ops, return
      | true to signal executor to schedule
      | a CPU op
      |
      */
    #[inline] pub fn is_stream_free(
        option:    &DeviceOption,
        stream_id: i32) -> bool 
    {
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn rand_generator(&mut self) -> *mut RandGenType {
        
        todo!();
        /*
            if (!random_generator_.get()) {
          random_generator_.reset(new rand_gen_type(RandSeed()));
        }
        return random_generator_.get();
        */
    }
    
    #[inline] pub fn rand_seed(&mut self) -> u32 {
        
        todo!();
        /*
            if (!random_seed_set_) {
          random_seed_ = RandomNumberSeed();
          random_seed_set_ = true;
        }
        return static_cast<uint32_t>(random_seed_);
        */
    }
    
    #[inline] pub fn switch_to_device(&mut self, stream_id: i32)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn wait_event(&mut self, ev: &Event)  {
        
        todo!();
        /*
            ev.Wait(CPU, this);
        */
    }
    
    #[inline] pub fn record(
        &self, 
        ev:      *mut Event,
        err_msg: *const u8)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(ev, "Event must not be null.");
        ev->Record(CPU, this, err_msg);
        */
    }
    
    #[inline] pub fn finish_device_computation(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn copy_bytes_same_device(
        &mut self, 
        nbytes: usize,
        src:    *const c_void,
        dst:    *mut c_void)  
    {
        todo!();
        /*
            CopyBytesImpl(nbytes, src, dst);
        */
    }

    #[inline] pub fn copy_bytes(
        nbytes: usize, 
        src:    *const c_void,
        dst:    *mut c_void) 
    {
        todo!();
        /*
          if (nbytes == 0) {
            return;
          }
          CAFFE_ENFORCE(src);
          CAFFE_ENFORCE(dst);
          memcpy(dst, src, nbytes);
        */
    }
}

/**
  | A function to generate a random number
  | seed that is unique in a best-effort
  | basis, using an ever-incrementing
  | seed and the current time.
  |
  */
#[inline] pub fn random_number_seed() -> u32 {
    
    todo!();
    /*
        // Originally copied from folly::randomNumberSeed (at 418ad4)
      // modified to use chrono instead of sys/time.h
      static std::atomic<uint32_t> seedInput(0);
      auto tv = std::chrono::system_clock::now().time_since_epoch();
      uint64_t usec = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(tv).count());
      uint32_t tv_sec = usec / 1000000;
      uint32_t tv_usec = usec % 1000000;
      const uint32_t kPrime0 = 51551;
      const uint32_t kPrime1 = 61631;
      const uint32_t kPrime2 = 64997;
      const uint32_t kPrime3 = 111857;
      return kPrime0 * (seedInput++) + kPrime1 * static_cast<uint32_t>(getpid()) +
          kPrime2 * tv_sec + kPrime3 * tv_usec;
    */
}

#[inline] pub fn copy_bytes_impl(
    nbytes: usize,
    src:    *const c_void,
    dst:    *mut c_void)  
{
    todo!();
    /*
        if (nbytes == 0) {
        return;
      }
      CAFFE_ENFORCE(src);
      CAFFE_ENFORCE(dst);
      memcpy(dst, src, nbytes);
    */
}

#[inline] pub fn copy_bytes_wrapper(
    nbytes:     usize,
    src:        *const c_void,
    src_device: Device,
    dst:        *mut c_void,
    dst_device: Device)  
{
    todo!();
    /*
        CopyBytesImpl(nbytes, src, dst);
    */
}

register_context!{
    DeviceType::CPU, 
    CPUContext
}

register_copy_bytes_function!{
    /*
    DeviceType::CPU,
    DeviceType::CPU,
    caffe2::CopyBytesWrapper
    */
}
