crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPInt8ConvOp {
    base:                       IDEEPConvPoolOpBase,
    attr:                       IAttr,
    algo:                       IAlgo,
    scale:                      f32,
    last_input:                 i32,
    zero_point:                 i32,
    lowp_kind:                  ILowpKind,
    fusion_type:                FusionType,
    filter:                     ITensor,
    bias:                       ITensor,
    y_scales:                   IScale,
    cached_X_descriptor:        ITensorDescriptor,
    cached_weights_descriptor:  ITensorDescriptor,
    conv_param:                 IDEEPConvolutionForwardParams,
}

input_tags!{
    IDEEPInt8ConvOp {
        InputX,
        Filter,
        BiasOrInputS,
        InputS
    }
}

output_tags!{
    IDEEPInt8ConvOp {
        Output
    }
}

impl IDEEPInt8ConvOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPConvPoolOpBase(operator_def, ws),
            scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
            zero_point_( this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) 

        OPERATOR_NEEDS_FEATURE(pad_l() == pad_r() && pad_t() == pad_b(),
                               "Uneven padding not supported.");
        fusion_type_ = FUSION_UNKNOWN;
        last_input_ = BIAS_OR_INPUT_S;
        algo_ = ialgo::convolution_direct;
        auto conv_algorithm = OperatorStorage::GetSingleArgument<int>(
            "conv_algorithm", CONV_ALGORITHM_AUTO);
        if (conv_algorithm == CONV_ALGORITHM_WINOGRAD) {
          algo_ = ialgo::convolution_winograd;
        }
        CAFFE_ENFORCE(zero_point_ == 128 || zero_point_ == 0);
        Y_scales_ = ConvertScales({scale_});
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto &X = Input(INPUT_X);
        const auto &filter = Input(FILTER);
        auto *Y = Output(OUTPUT);

        CAFFE_ENFORCE(X.has_scale());
        CAFFE_ENFORCE(4 == X.ndims() && 4 == filter.ndims());
        CAFFE_ENFORCE(X.get_data_type() == idtype::s8
            || X.get_data_type() == idtype::u8);
        CAFFE_ENFORCE(filter.get_dim(2) == kernel_h());
        CAFFE_ENFORCE(filter.get_dim(3) == kernel_w());
        CAFFE_ENFORCE(
            X.get_dim(1) == filter.get_dim(1) * group_,
            "Convolution op: input channels does not match: # of input channels ",
            X.get_dim(1), " is not equal to kernel channels * group:",
            filter.get_dim(1), "*", group_);

        bool input_changed = (cached_X_descriptor_ != X.get_descriptor());
        if (input_changed) {
          cached_X_descriptor_ = X.dup_descriptor();
        }

        bool weights_changed = (cached_weights_descriptor_ != filter.get_descriptor());
        if (weights_changed) {
          cached_weights_descriptor_ = filter.dup_descriptor();
          CAFFE_ENFORCE(filter.get_data_type() == idtype::s8 && filter.has_scale());

          auto X_dt = X.get_data_type();
          lowp_kind_ = ilowp_kind::LOWP_U8S8;
          if (X_dt == idtype::s8) {
            lowp_kind_ = ilowp_kind::LOWP_S8S8;
          }

          auto expected_descriptor =
              ideep::convolution_forward::expected_weights_desc(
                  filter.get_dims(),
                  idtype::s8,
                  {stride_.begin(), stride_.end()},
                  pad_tl(),
                  pad_br(),
                  {dilation_.begin(), dilation_.end()},
                  group_,
                  algo_,
                  iprop::forward_inference,
                  X_dt, X.get_dims());
          if (filter.get_desc() != expected_descriptor) {
            filter_.init(expected_descriptor);
            filter_.set_scale(filter.get_scale());
            filter_.feed_from(filter);
          } else {
            filter_ = filter;
          }

          if (InputSize() > last_input_) {
            // NOTE: If the bias is shared by other operators in this module,
            // The existing bias scale should not satisfy current operator.
            // Thus, we have to requantize it by current input and filter scales.
            auto bias = Input(BIAS_OR_INPUT_S);
            bias_.init({bias.get_dims(), idtype::s32});
            iscale bias_scales (filter_.get_scale());
            for (auto &scale : bias_scales) { scale *= X.get_scale()[0]; }
            bias_.set_scale(bias_scales);
            bias_.feed_from(bias);
          }
        }

        bool with_bias = InputSize() > last_input_;
        if (input_changed || weights_changed) {
          auto Y_dims = CalcOutputDims(X, filter.get_dim(0));
          if (with_bias) {
            ideep::convolution_forward::prepare(
                conv_param,
                X,
                filter_,
                bias_,
                Y_dims,
                *Y,
                {stride_.begin(), stride_.end()},
                {dilation_.begin(), dilation_.end()},
                pad_tl(),
                pad_br(),
                group_,
                iscale(),
                iscale(),
                Y_scales_,
                attr_,
                algo_,
                iprop::forward_inference,
                lowp_kind_);
          } else {
            ideep::convolution_forward::prepare(
                conv_param,
                X,
                filter_,
                Y_dims,
                *Y,
                {stride_.begin(), stride_.end()},
                {dilation_.begin(), dilation_.end()},
                pad_tl(),
                pad_br(),
                group_,
                iscale(),
                iscale(),
                Y_scales_,
                attr_,
                algo_,
                iprop::forward_inference,
                lowp_kind_);
          }
        }

        if (with_bias) {
          ideep::convolution_forward::compute(conv_param, X, filter_, bias_, *Y);
        } else {
          ideep::convolution_forward::compute(conv_param, X, filter_, *Y);
        }

        if (fusion_type_ != FUSION_CONV_RELU && fusion_type_ != FUSION_UNKNOWN) {
          CAFFE_ENFORCE(
              Y == &(Input(InputSize() - 1)),
              "Convolution fusion op: InPlace is enforced for sum fusion.");
        }

        return true;
        */
    }
}
