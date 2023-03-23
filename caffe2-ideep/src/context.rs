crate::ix!();

pub struct IDEEPContext {

    // TODO(jiayq): instead of hard-coding a generator, 
    // make it more flexible.
    random_seed: i32, //1701
    random_generator: Box<RandGenType>,
} 

type RandGenType = mt19937::MT19937;

impl Default for IDEEPContext {
    
    fn default() -> Self {
        todo!();
        /*
            : random_seed_(RandomNumberSeed()
        */
    }
}

impl From<&DeviceOption> for IDEEPContext {

    fn from(option: &DeviceOption) -> Self {
        todo!();
        /*
            : random_seed_( option.has_random_seed() ? option.random_seed() : RandomNumberSeed()) 

        CAFFE_ENFORCE_EQ(option.device_type(), PROTO_IDEEP);
        */
    }
}

impl From<&Device> for IDEEPContext {

    fn from(device: &Device) -> Self {
        todo!();
        /*
            : IDEEPContext(DeviceToOption(device))
        */
    }
}

impl IDEEPContext {

    #[inline] fn new(nbytes: usize) -> DataPtr {
        
        todo!();
        /*
            return GetAllocator(CPU)->allocate(nbytes);
        */
    }

    #[inline] fn copy<T, SrcContext, DstContext>(
        &mut self,
        n: usize,
        src: *const T,
        dst: *mut T) {
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

    #[inline] fn copy_items<SrcContext, DstContext>(
        &mut self, 
        meta: TypeMeta,
        n: usize,
        src: *const c_void,
        dst: *mut c_void) {
        todo!();
        /*
            if (meta.copy()) {
              meta.copy()(src, dst, n);
            } else {
              CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
            }
        */
    }
    

    #[inline] fn rand_generator(&mut self) -> &mut RandGenType {
        
        todo!();
        /*
            if (!random_generator_.get()) {
          random_generator_.reset(new rand_gen_type(random_seed_));
        }
        return *random_generator_.get();
        */
    }

    #[inline] fn has_async_part_default() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] fn supports_async_scheduling() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] fn is_stream_free(option: &DeviceOption, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] fn get_device_type() -> DeviceType {
        
        todo!();
        /*
            return IDEEP;
        */
    }
    
    /// Two copy functions that deals with cross-device copies.
    #[inline] fn copy_bytes<Context>(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void)  
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

impl BaseContext for IDEEPContext {
    
    #[inline] fn switch_to_device(&mut self, stream_id: StreamId)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] fn wait_event(&mut self, ev: &Event)  {
        
        todo!();
        /*
            ev.Wait(IDEEP, this);
        */
    }
    
    #[inline] fn record(&self, ev: *mut Event, err_msg: *const u8)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(ev, "Event must not be null.");
        ev->Record(IDEEP, this, err_msg);
        */
    }

    #[inline] fn finish_device_computation(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] fn copy_bytes_same_device(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void)  
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
    
    #[inline] fn copy_bytes_fromCPU(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void)  
    {
        todo!();
        /*
            CopyBytesSameDevice(nbytes, src, dst);
        */
    }
    
    #[inline] fn copy_bytes_toCPU(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void)  
    {
        todo!();
        /*
            CopyBytesSameDevice(nbytes, src, dst);
        */
    }
    
    #[inline] fn supports_non_fundamental_types(&self) -> bool {
        
        todo!();
        /*
            // IDEEP meta copy is OK
        return true;
        */
    }

    #[inline] fn device(&self) -> Device {
        
        todo!();
        /*
            return at::Device(IDEEP);
        */
    }
    
    #[inline] fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return IDEEP;
        */
    }
}
