crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/OpaqueTensorImpl.h]

/**
  | An "Opaque" TensorImpl -- there are no strides
  | and (for now) even data() is not supported
  | (thus no pointer arithmetic).
  |
  | NOTE: We could allow data() in the future, but
  | would have to ensure pointer arithmetic code is
  | properly guarded.
  |
  | NOTE: This does not support resize_ (and other
  | metadata-changing ops) because of
  | `shallow_copy_and_detach`. We would need to
  | define an interface to "shallow copy" in order
  | to add support.
  */
pub struct OpaqueTensorImpl<OpaqueHandle> {
    base:          TensorImpl,
    opaque_handle: OpaqueHandle,
}

impl OpaqueTensorImpl<OpaqueHandle> {

    /// public constructor for now...
    pub fn new(
        key_set:                      DispatchKeySet,
        data_type:                    TypeMeta,
        device:                       Device,
        opaque_handle:                OpaqueHandle,
        sizes:                        &[i32],
        is_non_overlapping_and_dense: bool) -> Self {
        let is_non_overlapping_and_dense: bool =
                 is_non_overlapping_and_dense.unwrap_or(true);
        todo!();
        /*


            : TensorImpl(key_set, data_type, device),
            opaque_handle_(move(opaque_handle)) 

        set_storage_access_should_throw();
        set_has_contiguity_policy(HasContiguityPolicy::ContiguityNotSupported);
        sizes_and_strides_.set_sizes(sizes);
        refresh_numel();
        is_non_overlapping_and_dense_ = is_non_overlapping_and_dense;
        */
    }
    
    pub fn release_resources(&mut self)  {
        
        todo!();
        /*
            TensorImpl::release_resources();
        opaque_handle_ = {};
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            AT_ERROR("opaque tensors do not have strides");
        */
    }
    
    pub fn stride(&self, d: i64) -> i64 {
        
        todo!();
        /*
            AT_ERROR("opaque tensors do not have strides");
        */
    }
    
    pub fn set_size(&mut self, 
        dim:      i64,
        new_size: i64)  {
        
        todo!();
        /*
            AT_ERROR("opaque tensors do not have set_size");
        */
    }
    
    pub fn set_stride(&mut self, 
        dim:        i64,
        new_stride: i64)  {
        
        todo!();
        /*
            AT_ERROR("opaque tensors do not have set_stride");
        */
    }
    
    pub fn set_storage_offset(&mut self, storage_offset: i64)  {
        
        todo!();
        /*
            AT_ERROR("opaque tensors do not have set_storage_offset");
        */
    }

    #[cfg(debug_assertions)]
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "OpaqueTensorImpl assumes that storage_ is never set");
        return false;
        */
    }

    /**
      | Return a TensorImpl that is a shallow-copy
      | of this TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn shallow_copy_and_detach(&self, 
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            auto impl = make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
            key_set(), dtype(), device(), opaque_handle_, sizes_and_strides_.sizes_arrayref());
        copy_tensor_metadata(
            /*src_opaque_impl=*/this,
            /*dest_opaque_impl=*/impl.get(),
            /*version_counter=*/version_counter,
            /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
        impl->refresh_numel();
        return impl;
        */
    }

    /**
      | Return a TensorImpl that is a shallow-copy
      | of this TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn shallow_copy_and_detach(&self, 
        version_counter:              VariableVersion,
        allow_tensor_metadata_change: bool) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            auto impl = make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
            key_set(), dtype(), device(), opaque_handle_, sizes_and_strides_.sizes_arrayref());
        copy_tensor_metadata(
            /*src_opaque_impl=*/this,
            /*dest_opaque_impl=*/impl.get(),
            /*version_counter=*/move(version_counter),
            /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
        impl->refresh_numel();
        return impl;
        */
    }

    /**
      | Shallow-copies data from another TensorImpl
      | into this TensorImpl.
      | 
      | For why this function doesn't check
      | this TensorImpl's `allow_tensor_metadata_change_`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn shallow_copy_from(&mut self, impl_: &IntrusivePtr<TensorImpl>)  {
        
        todo!();
        /*
            AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
        auto opaque_impl =
            static_cast<const OpaqueTensorImpl<OpaqueHandle>*>(impl.get());
        copy_tensor_metadata(
            /*src_impl=*/opaque_impl,
            /*dest_impl=*/this,
            /*version_counter=*/version_counter(),
            /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
        refresh_numel();
        */
    }
    
    pub fn opaque_handle(&self) -> &OpaqueHandle {
        
        todo!();
        /*
            return opaque_handle_;
        */
    }
    
    pub fn unsafe_opaque_handle(&mut self) -> &mut OpaqueHandle {
        
        todo!();
        /*
            return opaque_handle_;
        */
    }
 
    /**
      | Copy the tensor metadata fields (e.g.
      | sizes / strides / storage pointer / storage_offset)
      | from one TensorImpl to another TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn copy_tensor_metadata(
        src_opaque_impl:              *const OpaqueTensorImpl<OpaqueHandle>,
        dest_opaque_impl:             *mut OpaqueTensorImpl<OpaqueHandle>,
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
            TensorImpl::copy_tensor_metadata(
            src_opaque_impl,
            dest_opaque_impl,
            version_counter,
            allow_tensor_metadata_change);

        // OpaqueTensorImpl-specific fields.
        dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
        */
    }
    
    pub fn copy_tensor_metadata(
        src_opaque_impl:              *const OpaqueTensorImpl<OpaqueHandle>,
        dest_opaque_impl:             *mut OpaqueTensorImpl<OpaqueHandle>,
        version_counter:              VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
            TensorImpl::copy_tensor_metadata(
            src_opaque_impl,
            dest_opaque_impl,
            move(version_counter),
            allow_tensor_metadata_change);

        // OpaqueTensorImpl-specific fields.
        dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "OpaqueTensorImpl";
        */
    }
}
