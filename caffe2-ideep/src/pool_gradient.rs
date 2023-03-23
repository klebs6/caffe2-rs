crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPPoolGradientOp {
    base: IDEEPConvPoolOpBase,
    algo: IAlgo,
}

input_tags!{
    IDEEPPoolGradientOp {
        Input, 
        Output, 
        OutputGrad
    }
}

output_tags!{
    IDEEPPoolGradientOp {
        InputGrad
    }
}

impl IDEEPPoolGradientOp {
    
    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& Y = Input(OUTPUT);
        const auto& dY = Input(OUTPUT_GRAD);
        auto* dX = Output(INPUT_GRAD);

        ideep::pooling_backward::compute(dY, Y, X, *dX,
                                         {stride_.begin(), stride_.end()},
                                         {kernel_.begin(), kernel_.end()},
                                         pad_tl(), pad_br(), algo_);

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
        // Figure out the pooling descriptor.
        if (operator_def.type().substr(0, 15) == "MaxPoolGradient") {
          algo_ = ialgo::pooling_max;
        } else if (operator_def.type().substr(0, 19) == "AveragePoolGradient") {
          algo_ = ialgo::pooling_avg;
        } else {
          LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
        }
        */
    }
}

