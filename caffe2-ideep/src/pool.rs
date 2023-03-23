crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPPoolOp {
    base:                IDEEPConvPoolOpBase,
    pk:                  IProp,
    algo:                IAlgo,
    cached_x_descriptor: ITensorDescriptor,
}

input_tags!{
    IDEEPPoolOp {
        Input
    }
}

output_tags!{
    IDEEPPoolOp {
        Output
    }
}

impl IDEEPPoolOp {
    
    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
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
                                        pad_tl(), pad_br(), algo_, pk_);

        return true;
        */
    }
    
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

        bool training_mode = OperatorStorage::GetSingleArgument<int>("training_mode", 1);
        pk_ = training_mode ? iprop::forward_training : iprop::forward_inference;

        // Figure out the pooling descriptor.
        if (operator_def.type().substr(0, 7) == "MaxPool") {
          algo_ = ialgo::pooling_max;
        } else if (operator_def.type().substr(0, 11) == "AveragePool") {
          algo_ = ialgo::pooling_avg;
        } else {
          LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
        }
        */
    }
}

