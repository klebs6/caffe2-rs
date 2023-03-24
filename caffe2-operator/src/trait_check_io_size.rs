crate::ix!();

pub trait CheckInputSize {

    #[inline] fn input_size(&self) -> i32 {
        
        todo!();
        /*
            return input_size_;
        */
    }
}

pub trait CheckOutputSize {

    #[inline] fn output_size(&self) -> i32 {
        
        todo!();
        /*
            if (isLegacyOperator()) {
          return outputs_.size();
        }
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        return newstyle_outputs_.size();
    #else
        CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}

