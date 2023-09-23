crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/xla_tensor_test.cpp]

pub fn xla_free(ptr: *mut c_void)  {
    
    todo!();
        /*
      free(ptr);
        */
}

pub fn xla_malloc(size: libc::ptrdiff_t)  {
    
    todo!();
        /*
      return malloc(size);
        */
}

pub struct XLAAllocator {
    base: Allocator,
}

impl XLAAllocator {
    
    pub fn allocate(&self, size: usize) -> DataPtr {
        
        todo!();
        /*
            auto* ptr = XLAMalloc(size);
        return {ptr, ptr, &XLAFree, DeviceType_XLA};
        */
    }
    
    pub fn raw_deleter(&self) -> DeleterFnPtr {
        
        todo!();
        /*
            return &XLAFree;
        */
    }
}

#[test] fn xla_tensor_test_no_storage() {
    todo!();
    /*
    
      XLAAllocator allocator;
      auto tensor_impl = make_intrusive<TensorImpl, UndefinedTensorImpl>(
          DispatchKey::XLA,
          TypeMeta::Make<float>(),
          Device(DeviceType_XLA, 0));
      Tensor t(move(tensor_impl));
      ASSERT_TRUE(t.device() == Device(DeviceType_XLA, 0));

    */
}
