crate::ix!();


lazy_static!{

    /**
      | A global table that collects min/max for
      | each tensor name.
      |
      | Useful in case there are multiple copies
      | of the same network.
      */
    static ref min_max_map: HashMap<String, (f32, f32)> = HashMap::new();
}

#[inline] pub fn find_min_max<T>(
    data: *const T,
    min:  *mut f32,
    max:  *mut f32,
    len:  i32)  {

    todo!();
    /*
        vector<float> temp(len);
      for (int i = 0; i < len; ++i) {
        temp[i] = data[i];
      }
      fbgemm::FindMinMax(temp.data(), min, max, len);
    */
}

#[inline] pub fn get_float_tensor_data(tensor: *mut TensorCPU) -> *mut f32 {
    
    todo!();
    /*
        float* data = nullptr;
      vector<float> data_temp;
      if (tensor->IsType<float>()) {
        if (!tensor->data<float>()) {
          return nullptr;
        }
        data = tensor->template data<float>();
      } else if (tensor->IsType<int>()) {
        if (!tensor->data<int>()) {
          return nullptr;
        }
        const int* data_orig = tensor->data<int>();
        data_temp.resize(tensor->numel());
        for (int j = 0; j < tensor->numel(); ++j) {
          data_temp[j] = data_orig[j];
        }
        data = data_temp.data();
      } else if (tensor->IsType<long>()) {
        if (!tensor->data<long>()) {
          return nullptr;
        }
        const long* data_orig = tensor->data<long>();
        data_temp.resize(tensor->numel());
        for (int j = 0; j < tensor->numel(); ++j) {
          data_temp[j] = data_orig[j];
        }
        data = data_temp.data();
      } else {
        return nullptr;
      }
      return data;
    */
}


#[inline] pub fn find_min_maxf32(
    data: *const f32,
    min:  *mut f32,
    max:  *mut f32,
    len:  i32)  {
    
    todo!();
    /*
        fbgemm::FindMinMax(data, min, max, len);
    */
}

#[inline] pub fn has_dnn_lowp_engine_from_op_def(op_def: &OperatorDef) -> bool {
    
    todo!();
    /*
        const string ENGINE_PREFIX = "DNNLOWP";
      return strncmp(
                 op_def.engine().c_str(),
                 ENGINE_PREFIX.c_str(),
                 ENGINE_PREFIX.size()) == 0;
    */
}

#[inline] pub fn has_dnn_lowp_engine_from_op_base(op: &OperatorStorage) -> bool {
    
    todo!();
    /*
        return HasDNNLowPEngine_(op.debug_def());
    */
}
