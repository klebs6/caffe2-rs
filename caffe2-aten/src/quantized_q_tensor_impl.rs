crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/quantized/QTensorImpl.h]

/**
  | QTensorImpl is a TensorImpl for Quantized
  | Tensors, it stores Quantizer which
  | specifies the quantization scheme
  | and parameters, for more information
  | please see ATen/quantized/Quantizer.h
  | 
  | We'll use QTensor in code or documentation
  | to refer to a Tensor with QTensorImpl.
  |
  */
pub struct QTensorImpl {
    base:      TensorImpl,
    quantizer: QuantizerPtr,
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/quantized/QTensorImpl.cpp]
impl QTensorImpl {
    
    pub fn new(
        storage:   Storage,
        key_set:   DispatchKeySet,
        data_type: TypeMeta,
        quantizer: QuantizerPtr) -> Self {
    
        todo!();
        /*


        
        */
    }

    /// See Note [Enum ImplType]
    pub fn new(
        ty:        ImplType,
        storage:   Storage,
        key_set:   DispatchKeySet,
        data_type: TypeMeta,
        quantizer: QuantizerPtr) -> Self {
    
        todo!();
        /*


        
        */
    }

    /// TODO: Expose in PyTorch Frontend
    pub fn quantizer(&mut self) -> QuantizerPtr {
        
        todo!();
        /*
            return quantizer_;
        */
    }
    
    pub fn set_quantizer(&mut self, quantizer: QuantizerPtr)  {
        
        todo!();
        /*
            quantizer_ = quantizer;
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
            auto impl = make_intrusive<QTensorImpl>(
            Storage(storage()), key_set(), data_type_, quantizer_);
        copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/version_counter,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
        impl->refresh_numel();
        impl->refresh_contiguous();
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
            auto impl = make_intrusive<QTensorImpl>(
            Storage(storage()), key_set(), data_type_, quantizer_);
        copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/move(version_counter),
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
        impl->refresh_numel();
        impl->refresh_contiguous();
        return impl;
        */
    }

    /**
      | Shallow-copies data from another TensorImpl
      | into this TensorImpl.
      | 
      | For why this function doesn't check
      | this TensorImpl's `allow_tensor_metadata_change_`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn shallow_copy_from(&mut self, impl_: &IntrusivePtr<TensorImpl>)  {
        
        todo!();
        /*
            AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
        auto q_impl = static_cast<const QTensorImpl*>(impl.get());
        copy_tensor_metadata(
          /*src_impl=*/q_impl,
          /*dest_impl=*/this,
          /*version_counter=*/version_counter(),
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
        refresh_numel();
        refresh_contiguous();
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
        
        */
    }

    /**
      | Copy the tensor metadata fields (e.g.
      | sizes / strides / storage pointer / storage_offset)
      | from one TensorImpl to another TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn copy_tensor_metadata(
        src_q_impl:                   *const QTensorImpl,
        dest_q_impl:                  *mut QTensorImpl,
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
        TensorImpl::copy_tensor_metadata(src_q_impl, dest_q_impl, version_counter, allow_tensor_metadata_change);

        // OpaqueTensorImpl-specific fields.
        dest_q_impl->quantizer_ = src_q_impl->quantizer_;
        */
    }
    
    pub fn new(
        storage:   Storage,
        key_set:   DispatchKeySet,
        data_type: TypeMeta,
        quantizer: QuantizerPtr) -> Self {
    
        todo!();
        /*
        : TensorImpl(move(storage), key_set, data_type),
          quantizer_(quantizer)
        */
    }
    
    pub fn new(
        ty:        ImplType,
        storage:   Storage,
        key_set:   DispatchKeySet,
        data_type: TypeMeta,
        quantizer: QuantizerPtr) -> Self {
    
        todo!();
        /*
          : TensorImpl(type, move(storage), key_set, data_type),
          quantizer_(quantizer)
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "QTensorImpl";
        */
    }
}
