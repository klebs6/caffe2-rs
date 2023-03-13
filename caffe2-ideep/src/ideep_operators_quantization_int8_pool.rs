crate::ix!();


pub struct IDEEPInt8PoolOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();
    base: IDEEPConvPoolOpBase,

    algo: IAlgo,
    cached_x_descriptor: ITensorDescriptor,
}

input_tags!{
    IDEEPInt8PoolOp {
        Input
    }
}

output_tags!{
    IDEEPInt8PoolOp {
        Output
    }
}

impl IDEEPInt8PoolOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPConvPoolOpBase(operator_def, ws) 

        CAFFE_ENFORCE(
            (dilation_h() == 1) && (dilation_w() == 1),
            "Pooling op does not support dilation right now.");
        if (!global_pooling_) {
          CAFFE_ENFORCE(
              pad_t() < kernel_h() && pad_b() < kernel_h() &&
                  pad_l() < kernel_w() && pad_r() < kernel_w(),
              "Pad should be smaller than kernel.");
        }

        // Figure out the pooling descriptor.
        if (operator_def.type().substr(0, 11) == "Int8MaxPool") {
          algo_ = ialgo::pooling_max;
        } else if (operator_def.type().substr(0, 15) == "Int8AveragePool") {
          algo_ = ialgo::pooling_avg;
        } else {
          LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);
        auto Y_dims = CalcOutputDims(X, X.get_dim(1));

        if (cached_X_descriptor_ != X.get_descriptor()) {
          cached_X_descriptor_ = X.dup_descriptor();
        }

        ideep::pooling_forward::compute(X, Y_dims, *Y,
                                        {stride_.begin(), stride_.end()},
                                        {kernel_.begin(), kernel_.end()},
                                        pad_tl(), pad_br(), algo_,
                                        iprop::forward_inference);

        return true;
        */
    }
}

register_ideep_operator_with_engine!{Int8MaxPool,     DNNLOWP, IDEEPInt8PoolOp}
register_ideep_operator_with_engine!{Int8AveragePool, DNNLOWP, IDEEPInt8PoolOp}
