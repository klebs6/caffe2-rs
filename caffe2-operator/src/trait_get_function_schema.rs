crate::ix!();

pub trait GetFunctionSchema {

    #[inline] fn get_function_schema(&self) -> &FunctionSchema {
        
        todo!();
        /*
            CAFFE_ENFORCE(!isLegacyOperator());
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        return *fn_schema_.get();
    #else
        CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}
