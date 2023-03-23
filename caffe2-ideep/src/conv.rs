crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPConvOp {
    base:                           IDEEPConvPoolOpBase,
    pk:                             IProp,
    algo:                           IAlgo,
    attr:                           IAttr,
    last_input:                     i32,
    training_mode:                  bool,
    fusion_type:                    FusionType,
    filter:                         ITensor,
    dummy_scale:                    IScale,
    cached_X_descriptor:            ITensorDescriptor,
    cached_weights_descriptor:      ITensorDescriptor,
    conv_param:                     IDEEPConvolutionForwardParams,
}

input_tags!{
    IDEEPConvOp {
        InputX,
        Filter,
        BiasOrInputS,
        InputS
    }
}

output_tags!{
    IDEEPConvOp {
        Output
    }
}

impl IDEEPConvOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPConvPoolOpBase(operator_def, ws) 

        OPERATOR_NEEDS_FEATURE(
            order_ == StorageOrder::NCHW, "Unsupported storage order.");
        OPERATOR_NEEDS_FEATURE(
            pad_l() == pad_r() && pad_t() == pad_b(),
            "Uneven padding not supported.");

        fusion_type_ = FUSION_UNKNOWN;
        last_input_ = BIAS_OR_INPUT_S;

        training_mode_ = OperatorStorage::GetSingleArgument<int>("training_mode", 0);
        pk_ = training_mode_ ? iprop::forward_training : iprop::forward_inference;

        algo_ = ialgo::convolution_direct;
        auto conv_algorithm = OperatorStorage::GetSingleArgument<int>(
            "conv_algorithm", CONV_ALGORITHM_AUTO);
        if (conv_algorithm == CONV_ALGORITHM_WINOGRAD) {
          algo_ = ialgo::convolution_winograd;
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT_X);
        const auto& filter = Input(FILTER);
        auto* Y = Output(OUTPUT);

        CAFFE_ENFORCE(4 == X.ndims());
        CAFFE_ENFORCE(4 == filter.ndims());
        CAFFE_ENFORCE_EQ(filter.get_dim(2), kernel_h());
        CAFFE_ENFORCE_EQ(filter.get_dim(3), kernel_w());
        CAFFE_ENFORCE(
            X.get_dim(1) == filter.get_dim(1) * group_,
            "Convolution op: input channels does not match: # of input channels ",
            X.get_dim(1),
            " is not equal to kernel channels * group:",
            filter.get_dim(1),
            "*",
            group_);

        bool input_changed = (cached_X_descriptor_ != X.get_descriptor());
        if (input_changed) {
          cached_X_descriptor_ = X.dup_descriptor();
        }

        bool weights_changed = (cached_weights_descriptor_ != filter.get_descriptor());
        if (!training_mode_ && weights_changed) {
          cached_weights_descriptor_ = filter.dup_descriptor();
          auto expected_descriptor =
              ideep::convolution_forward::expected_weights_desc(
                  filter.get_dims(),
                  idtype::f32,
                  {stride_.begin(), stride_.end()},
                  pad_tl(),
                  pad_br(),
                  {dilation_.begin(), dilation_.end()},
                  group_,
                  algo_,
                  pk_,
                  idtype::f32,
                  X.get_dims());
          if (filter.get_descriptor() != expected_descriptor) {
            filter_.init(expected_descriptor);
            filter_.feed_from(filter);
          } else {
            filter_ = filter;
          }
        }

        bool with_bias = InputSize() > last_input_;
        auto filter_in = training_mode_ ? filter : filter_;
        if (training_mode_ || input_changed || weights_changed) {
          auto Y_dims_conv = CalcOutputDims(X, filter.get_dim(0));
          if (with_bias) {
            ideep::convolution_forward::prepare(
                conv_param,
                X,
                filter_in,
                Input(BIAS_OR_INPUT_S),
                Y_dims_conv,
                *Y,
                {stride_.begin(), stride_.end()},
                {dilation_.begin(), dilation_.end()},
                pad_tl(),
                pad_br(),
                group_,
                dummy_scale_,
                dummy_scale_,
                dummy_scale_,
                attr_,
                algo_,
                pk_);
          } else {
              ideep::convolution_forward::prepare(
                conv_param,
                X,
                filter_in,
                Y_dims_conv,
                *Y,
                {stride_.begin(), stride_.end()},
                {dilation_.begin(), dilation_.end()},
                pad_tl(),
                pad_br(),
                group_,
                dummy_scale_,
                dummy_scale_,
                dummy_scale_,
                attr_,
                algo_,
                pk_);
          }
        }

        if (with_bias) {
          ideep::convolution_forward::compute(conv_param, X, filter_in,
                                              Input(BIAS_OR_INPUT_S), *Y);
        } else {
          ideep::convolution_forward::compute(conv_param, X, filter_in, *Y);
        }

        if (fusion_type_ == FUSION_CONV_SUM
            && fusion_type_ == FUSION_CONV_SUM_RELU) {
          CAFFE_ENFORCE_EQ(Y,  &(Input(InputSize() - 1)),
              "Convolution fusion op: InPlace is enforced for sum fusion.");
        }

        return true;
        */
    }
}

