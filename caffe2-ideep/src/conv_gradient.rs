crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPConvGradientOp {
    base:    IDEEPConvPoolOpBase,
    no_bias: bool,
}

input_tags!{
    IDEEPConvGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    IDEEPConvGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl IDEEPConvGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPConvPoolOpBase(operator_def, ws),
            no_bias_(OperatorStorage::GetSingleArgument<int>("no_bias", 0)) 

        OPERATOR_NEEDS_FEATURE(
            pad_l() == pad_r() && pad_t() == pad_b(),
            "Uneven padding not supported.");
        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");
        CAFFE_ENFORCE(
            OperatorStorage::GetSingleArgument<int>("training_mode", 0),
            "In order to backward propagate weights correctly, "
            "please set training_mode=1");
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& filter = Input(FILTER);
        const auto& dY = Input(OUTPUT_GRAD);
        auto* dfilter = Output(FILTER_GRAD);

        if (no_bias_) {
          ideep::convolution_backward_weights::compute(
              X,
              dY,
              filter.get_dims(),
              *dfilter,
              {stride_.begin(), stride_.end()},
              {dilation_.begin(), dilation_.end()},
              pad_tl(),
              pad_br(),
              group_);
        } else {
          auto* dbias = Output(BIAS_OR_INPUT_GRAD);
          ideep::convolution_backward_weights::compute(
              X,
              dY,
              filter.get_dims(),
              *dfilter,
              *dbias,
              {stride_.begin(), stride_.end()},
              {dilation_.begin(), dilation_.end()},
              pad_tl(),
              pad_br(),
              group_);
        }

        if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
          auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
          ideep::convolution_backward_data::compute(
              dY,
              filter,
              X.get_dims(),
              *dX,
              {stride_.begin(), stride_.end()},
              {dilation_.begin(), dilation_.end()},
              pad_tl(),
              pad_br(),
              group_);
        }

        return true;
        */
    }
}
