crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanOpaqueTensorImpl.h]

/**
  | The only difference from OpaqueTensorImpl is
  | faking strides(), stride(), is_contiguous().
  |
  | The main intention for this is to be able to
  | run torchscript model on Vulkan backend.
  |
  | Strides are not supported on Vulkan side, plan
  | to support them.
  |
  */
pub struct VulkanOpaqueTensorImpl<OpaqueHandle> {
    base:    OpaqueTensorImpl<OpaqueHandle>,
    strides: SmallVec<[i64;5]>,
}

impl VulkanOpaqueTensorImpl<OpaqueHandle> {
    
    pub fn new(
        key_set:       DispatchKeySet,
        data_type:     TypeMeta,
        device:        Device,
        opaque_handle: OpaqueHandle,
        sizes:         &[i32],
        strides:       &[i32]) -> Self {
    
        todo!();
        /*


            : OpaqueTensorImpl<OpaqueHandle>(
                key_set,
                data_type,
                device,
                opaque_handle,
                sizes,
                false),
            strides_(strides.vec()) 

        TensorImpl::set_has_contiguity_policy(TensorImpl::HasContiguityPolicy::CustomBehavior);
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    pub fn is_contiguous_custom(&self, memory_format: MemoryFormat) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn stride(&self, d: i64) -> i64 {
        
        todo!();
        /*
            d = maybe_wrap_dim(d, this->dim(), false);
        return strides_[d];
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "VulkanOpaqueTensorImpl";
        */
    }
}
