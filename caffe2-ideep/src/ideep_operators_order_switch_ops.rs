crate::ix!();


pub struct IDEEPNHWC2NCHWOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,
}

input_tags!{
    IDEEPNHWC2NCHWOp {
        Input
    }
}

output_tags!{
    IDEEPNHWC2NCHWOp {
        Output
    }
}

impl IDEEPNHWC2NCHWOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        CAFFE_ENFORCE_EQ(X.ndims(), 4);
        CAFFE_ENFORCE(X.get_desc().is_nhwc());

        auto *Y = Output(OUTPUT);
        CAFFE_ENFORCE(Y != &X);

        // NOTE: NHWC changes the shape in framework, but not in MKL-DNN
        // Thus, for iDEEP tensor, the shapes of NCHW and NHWC are identical.
        Y->init({X.get_dims(), X.get_data_type(), iformat::nchw});
        Y->feed_from(X);
        return true;
        */
    }
}

///----------------------------------
pub struct IDEEPNCHW2NHWCOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,
}

input_tags!{
    IDEEPNCHW2NHWCOp {
        Input
    }
}

output_tags!{
    IDEEPNCHW2NHWCOp {
        Output
    }
}

impl IDEEPNCHW2NHWCOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        CAFFE_ENFORCE_EQ(X.ndims(), 4);
        CAFFE_ENFORCE(X.get_desc().is_nchw());

        auto *Y = Output(OUTPUT);
        CAFFE_ENFORCE(Y != &X);

        // NOTE: NHWC changes the shape in framework, but not in MKL-DNN
        // Thus, for iDEEP tensor, the shapes of NCHW and NHWC are identical.
        Y->init({X.get_dims(), X.get_data_type(), iformat::nhwc});
        Y->feed_from(X);
        return true;
        */
    }
}

register_ideep_operator!{NHWC2NCHW, IDEEPNHWC2NCHWOp}
register_ideep_operator!{NCHW2NHWC, IDEEPNCHW2NHWCOp}
