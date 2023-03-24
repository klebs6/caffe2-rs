crate::ix!();

pub trait GetRepeatedArgument {

    #[inline] fn get_repeated_argument<T>(
        &self,
        name: &String,
        default_value: &Vec<T>) -> Vec<T> 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
            CAFFE_ENFORCE(operator_def_, "operator_def was null!");
            return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(
                *operator_def_, name, default_value);
          }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
          auto index = argumentIndexWithName(name);
          CAFFE_ENFORCE(index.has_value(), "Couldn't get index for argument!", name);
          const auto& value = newstyle_inputs_[index.value()];
          return GetVectorFromIValueList<T>(value);
        #else
          CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }

    /**
      | We need this specialisation because IValue
      | based lists don't support int16_t. We need
      | to load it as List<int64_t> and transform
      | to int16_t.
      */
    #[inline] fn get_repeated_argumenti16(&self, name: &String, default_value: &Vec<i16>) -> Vec<i16> {
        
        todo!();
        /*
            if (isLegacyOperator()) {
        CAFFE_ENFORCE(operator_def_, "operator_def was null!");
        return ArgumentHelper::GetRepeatedArgument<OperatorDef, int16_t>(
            *operator_def_, name, default_value);
      }
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      auto index = argumentIndexWithName(name);
      CAFFE_ENFORCE(index.has_value(), "Couldn't get index for argument!", name);
      const auto& value = newstyle_inputs_[index.value()];
      auto vec = GetVectorFromIValueList<int64_t>(value);
      std::vector<int16_t> result;
      result.reserve(vec.size());
      for (int64_t elem : vec) {
        result.push_back(static_cast<int16_t>(elem));
      }
      return result;
    #else
      CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}
