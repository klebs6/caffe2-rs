crate::ix!();

#[inline] pub fn load_int_8tensor_info_of_blob(
    scale:  *mut Vec<f32>,
    offset: *mut Vec<f32>,
    axis:   *mut u32,
    b:      *const Blob)  
{
    todo!();
    /*
        const int8::Int8TensorCPU* int8_tensor =
          static_cast<const int8::Int8TensorCPU*>(b->GetRaw());
      scale->clear();
      offset->clear();
      scale->push_back(int8_tensor->scale);
      offset->push_back(int8_tensor->zero_point);
      *axis = 1;
    */
}

#[inline] pub fn get_tensor_shape_of_blob(b: *const Blob) -> TensorShape {
    
    todo!();
    /*
        TensorShape tp;
    #ifndef C10_MOBILE
      auto function_ptr =
          ExternalTensorFunctionsBaseRegistry()->Create(b->meta().id());
      if (function_ptr != nullptr) {
        // This is dnnlowp tensor and we cant deal with it using regular path
        auto dtype = function_ptr->GetExternalTensorType(b->GetRaw());
        tp.set_data_type(TypeMetaToDataType(dtype));

        size_t _capacity;
        DeviceOption _device;
        auto dshape =
            function_ptr->GetExternalTensorInfo(b->GetRaw(), &_capacity, &_device);
        for (auto d : dshape) {
          tp.add_dims(d);
        }
        return tp;
      }
    #endif

      TypeCall type_fun = GetTypeCallFunction(b->meta().id());
      TensorInfoCall tensor_info_fun = GetTensorInfoFunction(b->meta().id());
      if (type_fun) {
        tp.set_data_type(TypeMetaToDataType(type_fun(b->GetRaw())));
      }
      if (tensor_info_fun) {
        size_t _capacity;
        DeviceOption _device;
        auto shape = tensor_info_fun(b->GetRaw(), &_capacity, &_device);
        for (auto d : shape) {
          tp.add_dims(d);
        }
      } else {
        tp.set_unknown_shape(true);
      }
      return tp;
    */
}

