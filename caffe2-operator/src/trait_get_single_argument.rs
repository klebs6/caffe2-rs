crate::ix!();

pub trait GetSingleArgument {

    /**
      | Functions that deal with
      | arguments. Basically, this allows us to
      | map an argument name to a specific type of
      | argument that we are trying to access.
      */
    #[inline] fn get_single_argument<T>(name: &String, default_value: &T) -> T {
        todo!();
        /*
            if (isLegacyOperator()) {
              CAFFE_ENFORCE(operator_def_, "operator_def was null!");
              return ArgumentHelper::GetSingleArgument<OperatorDef, T>(
                  *operator_def_, name, default_value);
            }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
            auto index = argumentIndexWithName(name);
            CAFFE_ENFORCE(index.has_value(), "Couldn't get index for argument!", name);
            const auto& value = newstyle_inputs_[index.value()];
            return value.template to<T>();
        #else
            CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }

    #[inline] fn get_single_argument_net_def(
        &self, 
        name:          &String, 
        default_value: &NetDef) -> NetDef 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
        CAFFE_ENFORCE(operator_def_, "operator_def was null!");
        return ArgumentHelper::GetSingleArgument<OperatorDef, NetDef>(
            *operator_def_, name, default_value);
      }
      CAFFE_THROW("Cannot get NetDefs from IValue");
      return NetDef();
        */
    }
}
