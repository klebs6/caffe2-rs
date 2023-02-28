crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/UndefinedTensorImpl.h]
//-------------------------------------------[.cpp/pytorch/c10/core/UndefinedTensorImpl.cpp]

pub struct UndefinedTensorImpl {
    base: TensorImpl,
}

lazy_static!{
    /*
    static UndefinedTensorImpl _singleton;
    UndefinedTensorImpl UndefinedTensorImpl::_singleton;
    */
}
impl UndefinedTensorImpl {

    /**
      | Without this, we get:
      |
      |  error: identifier
      |    "UndefinedTensorImpl::_singleton" is
      |    undefined in device code
      |
      | (ostensibly because the constexpr tricks MSVC
      |   into trying to compile this function for
      |   device as well).
      |
      */
    #[inline] pub fn singleton() -> *mut TensorImpl {
        
        todo!();
        /*
            return &_singleton;
        */
    }
    
    /**
      | should this use the globalContext?
      | Can it get a context passed in somehow?
      |
      */
    pub fn new() -> Self {
    
        todo!();
        /*


            : TensorImpl(DispatchKey::Undefined, TypeMeta(), nullopt) 

      set_storage_access_should_throw();
        */
    }
    
    pub fn size(&self, d: i64) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(false, "size(dim) called on an undefined Tensor");
        */
    }
    
    pub fn stride(&self, d: i64) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(false, "stride(dim) called on an undefined Tensor");
        */
    }

    #[cfg(DEBUG)]
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          !storage_, "UndefinedTensorImpl assumes that storage_ is never set");
      return false;
        */
    }
    
    pub fn set_storage_offset(&mut self, _0: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "set_storage_offset() called on an undefined Tensor");
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            TORCH_CHECK(false, "strides() called on undefined Tensor");
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "UndefinedTensorImpl";
        */
    }
}
