crate::ix!();

#[inline] pub fn blob_is_int_8tensor_cputype(blob: &Blob) -> bool {
    
    todo!();
    /*
        return blob.meta().Match<int8::Int8TensorCPU>();
    */
}

#[inline] pub fn blob_is_tensor_type(
    blob: &Blob, 
    device_type: DeviceType) -> bool 
{
    todo!();

    /*
    let is_match: bool = blob.meta().r#match::<Tensor>();

    if !is_match {
        return false;
    }

    let tensor: *const Tensor = blob.get::<Tensor>();

    tensor != std::ptr::null() && (*tensor).get_device_type() == device_type
    */
}

#[inline] pub fn blob_set_tensor(
    blob: *mut Blob,
    tensor: Tensor) -> *mut Tensor 
{
    todo!();
    /*
        return blob->Reset<Tensor>(new Tensor(std::move(tensor)));
    */
}

#[inline] pub fn get_sized_tensor_with_options(
    previous_tensor: Tensor,
    dims:            &[i32],
    options:         TensorOptions) -> Tensor {

    todo!();
    /*
      Tensor tensor = std::move(previous_tensor);
      if (!tensor.defined()) {
        return caffe2::empty(dims, options);
      }
      if (tensor.GetDevice() == options.device() ||
          (!tensor.GetDevice().has_index() &&
           tensor.GetDeviceType() == options.device().type())) {
        if (tensor.sizes() != dims) {
          // Resize when the dims doesn't match
          tensor.Resize(dims);
        }
        if (tensor.dtype() == options.dtype()) {
          tensor.raw_mutable_data();
        } else {
          // create a new Tensor when the data_type doesn't match
          return caffe2::empty(dims, options);
        }
        return tensor;
      }
      return caffe2::empty(dims, options);
    */
}

/**
  | need to keep both functions that returns
  | Tensor* and the one returns Tensor for
  | clangr codemod
  |
  */
#[inline] pub fn blob_get_mutable_tensor_with_options(
    blob:    *mut Blob,
    dims:    &[i32],
    options: TensorOptions) -> *mut Tensor {
    
    todo!();
    /*
        if (blob->IsType<Tensor>()) {
        Tensor* tensor = blob->GetMutable<Tensor>();
        if (*tensor) {
          // We only compare device_type if the index is not set since there are Tensors
          // TODO: remove the extra check when all the Tensors are properly initialized
          if (tensor->GetDevice() == options.device() || (!tensor->GetDevice().has_index() && tensor->GetDeviceType() == options.device().type())) {
            if (tensor->sizes() != dims) {
              // Resize when the dims doesn't match
              tensor->Resize(dims);
            }
            if (tensor->dtype() == options.dtype()) {
              tensor->raw_mutable_data();
            } else {
              tensor->raw_mutable_data(options.dtype());
            }
            return tensor;
          }
          // create a new Tensor when device doesn't match
        }
      }

      VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
              << " dims: " << dims;
      // << " options: " << options; (operator<< for Options is in at:: now)
      return BlobSetTensor(blob, caffe2::empty(dims, options));
    */
}

#[inline] pub fn xBlob_get_mutable_tensor(
    blob:    *mut Blob,
    dims:    &[i32],
    options: TensorOptions) -> Tensor {

    todo!();
    /*
        return BlobGetMutableTensor(blob, dims, options)->UnsafeSharedInstance();
    */
}

#[inline] pub fn blob_get_mutable_tensor(
    blob:        *mut Blob,
    device_type: DeviceType) -> *mut Tensor {

    todo!();
    /*
      if (blob->IsType<Tensor>()) {
        Tensor* tensor = blob->GetMutable<Tensor>();
        if (*tensor && tensor->GetDeviceType() == device_type) {
          return tensor;
        }
      }

      // if we're here, then either Blob didn't hold a Tensor
      // or that Tensor had the wrong DeviceType.
      VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
              << " DeviceType:" << device_type;

      return BlobSetTensor(blob, Tensor(device_type));
    */
}

#[inline] pub fn blob_get_tensor(blob: &Blob,
    device_type: DeviceType) -> &Tensor {

    todo!();
    /*
      if (blob.IsType<Tensor>()) {
        const auto& tensor = blob.Get<Tensor>();
        if (tensor.GetDeviceType() == device_type) {
          return tensor;
        }
      }
      CAFFE_THROW("Blob didn't contain a Tensor or the device_type doesn't match");
    */
}

#[inline] pub fn blob_get_tensor_or_undefined(blob: &Blob) -> Tensor {
    
    todo!();
    /*
      if (blob.IsType<Tensor>()) {
        return blob.Get<Tensor>().UnsafeSharedInstance();
      } else {
        return Tensor();
      }
    */
}
