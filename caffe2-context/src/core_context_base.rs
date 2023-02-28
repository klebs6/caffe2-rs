crate::ix!();

/**
  | Virtual interface for the Context class
  | in Caffe2.
  | 
  | A Context defines all the necessities
  | to run an operator on a specific device.
  | Specific Context classes needs to implement
  | all the pure virtual functions in the
  | BaseContext class.
  | 
  | TODO: add docs after this is finalized.
  |
  */
pub trait BaseContext {

    fn device(&self) -> Device;
    fn device_type(&self) -> DeviceType;
    fn switch_to_device(&mut self, stream_id: StreamId);

    fn switch_to_device_0(&mut self)  {
        
        todo!();
        /*
            SwitchToDevice(0);
        */
    }
    
    fn wait_event(&mut self, ev: &Event);
    fn record(&self, ev: *mut Event, err_msg: *const u8);
    fn finish_device_computation(&mut self);
    
    /**
      | This used to be arbitrary cross-device
      | copy, but it turns out everyone did direct
      | CPU-X copy, so we just make three
      | functions for it (to avoid double
      | dispatch).  This will get obsoleted by
      | C10. where copies will be proper operators
      | (and get to rely on multiple dispatch
      | there.)
      */
    fn copy_bytes_same_device(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void);
    
    fn copy_bytes_fromCPU(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void);
    
    fn copy_bytes_toCPU(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void);

    fn copy_same_device<T>(&mut self, n: usize, src: *const T, dst: *mut T) {
        todo!();
        /*
            static_assert(
                c10::guts::is_fundamental<T>::value,
                "CopySameDevice requires fundamental types");
            CopyBytesSameDevice(
                n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
        */
    }

    fn copy_fromCPU<T>(&mut self, n: usize, src: *const T, dst: *mut T) {
        todo!();
        /*
            static_assert(
                c10::guts::is_fundamental<T>::value,
                "CopyFromCPU requires fundamental types");
            CopyBytesFromCPU(
                n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
        */
    }


    fn copy_toCPU<T>(&mut self, n: usize, src: *const T, dst: *mut T) {
        todo!();
        /*
            static_assert(
                c10::guts::is_fundamental<T>::value, "CopyToCPU requires fundamental types");
            CopyBytesToCPU(
                n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
        */
    }

    
    fn supports_non_fundamental_types(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn enforce_meta_copyOK(&mut self)  {
        
        todo!();
        /*
            AT_ASSERTM(
            SupportsNonFundamentalTypes(), "Context requires fundamental types");
        */
    }
    
    fn copy_items_same_device(
        &mut self, 
        meta: TypeMeta,
        n: usize,
        src: *const c_void,
        dst: *mut c_void)  
    {
        todo!();
        /*
            if (meta.copy()) {
          EnforceMetaCopyOK();
          meta.copy()(src, dst, n);
        } else {
          CopyBytesSameDevice(n * meta.itemsize(), src, dst);
        }
        */
    }
    
    fn copy_items_fromCPU(
        &mut self, 
        meta: TypeMeta,
        n: usize,
        src: *const c_void,
        dst: *mut c_void)  
    {
        todo!();
        /*
            if (meta.copy()) {
          EnforceMetaCopyOK();
          meta.copy()(src, dst, n);
        } else {
          CopyBytesFromCPU(n * meta.itemsize(), src, dst);
        }
        */
    }
    
    fn copy_items_toCPU(
        &mut self, 
        meta: TypeMeta,
        n:    usize,
        src:  *const c_void,
        dst:  *mut c_void)  
    {
        todo!();
        /*
            if (meta.copy()) {
          EnforceMetaCopyOK();
          meta.copy()(src, dst, n);
        } else {
          CopyBytesToCPU(n * meta.itemsize(), src, dst);
        }
        */
    }
}

// Context constructor registry
declare_typed_registry!{
    ContextRegistry,
    DeviceType,
    BaseContext,
    Box,
    Device
}

#[inline] pub fn create_context(device: &Device) //TODO -> Box<impl BaseContext> 
{
    
    todo!();
    /*
        return at::ContextRegistry()->Create(device.type(), device);
    */
}

define_typed_registry!{
    ContextRegistry,
    DeviceType,
    BaseContext,
    Box,
    Device
}
