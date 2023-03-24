crate::ix!();

pub trait CheckArgumentWithName {

    /**
      | -----------
      | @brief
      | 
      | Checks if the operator has an argument
      | of the given name.
      |
      */
    #[inline] fn has_argument(&self, name: &String) -> bool {
        
        todo!();
        /*
            if (isLegacyOperator()) {
          CAFFE_ENFORCE(operator_def_, "operator_def was null!");
          return ArgumentHelper::HasArgument(*operator_def_, name);
        }
        return argumentIndexWithName(name).has_value();
        */
    }

    #[inline] fn argument_index_with_name(&self, name: &String) -> Option<i32> {
        
        todo!();
        /*
            #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      return getFunctionSchema().argumentIndexWithName(name);
    #else
      CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}

pub trait CheckHasSingleArgumentOfType {

    #[inline] fn has_single_argument_of_type<T>(name: &String) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(operator_def_, "operator_def was null!");
            return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(
                *operator_def_, name);
        */
    }
}

