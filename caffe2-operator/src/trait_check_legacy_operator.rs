crate::ix!();

pub trait CheckLegacyOperator {

    /**
      | -----------
      | @brief
      | 
      | Return true if the operator was instantiated
      | with OperatorDef
      | 
      | New operators should be instantiated
      | with
      | 
      | FunctionSchema
      |
      */
    #[inline] fn is_legacy_operator(&self) -> bool {
        
        todo!();
        /*
            #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        return !fn_schema_;
    #else
        return true;
    #endif
        */
    }
}

