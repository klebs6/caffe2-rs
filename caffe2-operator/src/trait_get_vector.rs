crate::ix!();

pub trait GetVectorFromIValueList {

    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_from_ivalue_list<T>(value: &IValue) -> Vec<T> {
        todo!();
        /*
            return value.template to<List<T>>().vec();
        */
    }

    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_listi32(&self, value: &IValue) -> Vec<i32> {
        
        todo!();
        /*
            auto vs = value.toIntVector();
      vector<int> out;
      out.reserve(vs.size());
      for (int64_t v : vs) {
        out.emplace_back(v);
      }
      return out;
        */
    }
    
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_listf32(&self, value: &IValue) -> Vec<f32> {
        
        todo!();
        /*
            const auto& vs = value.toDoubleVector();
      vector<float> out;
      out.reserve(vs.size());
      for (double v : vs) {
        out.emplace_back(v);
      }
      return out;
        */
    }
    
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_list_string(&self, value: &IValue) -> Vec<String> {
        
        todo!();
        /*
            auto vs = value.template to<c10::List<string>>();
      vector<string> out;
      out.reserve(vs.size());
      for (string v : vs) {
        out.emplace_back(v);
      }
      return out;
        */
    }
    
    /**
      | We need this specialisation because IValue
      | based lists don't support int16_t. We need
      | to load it as List<int64_t> and transform
      | to int16_t.
      */
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_listi16(&self, value: &IValue) -> Vec<i16> {
        
        todo!();
        /*
            auto list = value.template to<c10::List<int64_t>>();
      std::vector<int16_t> result;
      result.reserve(list.size());
      for (int64_t elem : list) {
        result.push_back(static_cast<int16_t>(elem));
      }
      return result;
        */
    }
}

