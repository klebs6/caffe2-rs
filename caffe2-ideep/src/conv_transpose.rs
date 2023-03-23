crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS]
pub struct IDEEPConvTransposeOp {
    base:                      IDEEPConvTransposeUnpoolBase,
    training_mode:             bool,
    filter:                    IDEEPTensor,
    cached_weights_descriptor: IDEEPTensorDescriptor,
}

input_tags!{
    IDEEPConvTransposeOp {
        Input,
        Filter,
        Bias
    }
}

output_tags!{
    IDEEPConvTransposeOp {
        Output
    }
}

impl IDEEPConvTransposeOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPConvTransposeUnpoolBase(operator_def, ws),
            training_mode_( OperatorStorage::GetSingleArgument<int>("training_mode", 0)) 

        OPERATOR_NEEDS_FEATURE(
            pad_l() == pad_r() && pad_t() == pad_b(),
            "Uneven padding not supported.");
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& filter = Input(FILTER);
        auto* Y = Output(OUTPUT);
        CAFFE_ENFORCE_EQ(X.ndims(), 4);
        CAFFE_ENFORCE_EQ(filter.ndims(), 4);
        CAFFE_ENFORCE_EQ(filter.get_dim(2), kernel_h());
        CAFFE_ENFORCE_EQ(filter.get_dim(3), kernel_w());
        CAFFE_ENFORCE_EQ(filter.get_dim(0), X.get_dim(1),
                         "filter number must be equal to input channel number");

        auto Y_dims = CalcOutputDims(X, filter.get_dim(1));

        bool weights_changed = (cached_weights_descriptor_ != filter.get_descriptor());
        if (!training_mode_ && weights_changed) {
          cached_weights_descriptor_ = filter.dup_descriptor();
          auto filter_in = filter;

          auto expected_descriptor =
              ideep::convolution_transpose_forward::expected_weights_desc(
                  filter.get_dims(),
                  filter.get_data_type(),
                  {stride_.begin(), stride_.end()},
                  pad_tl(),
                  pad_br());
          if (filter_in.get_descriptor() != expected_descriptor) {
            filter_.init(expected_descriptor);
            filter_.feed_from(filter_in, /*is_deconv_weights=*/true);
          } else {
            filter_ = filter_in;
          }
        }

        auto transposed_filter = training_mode_ ? filter : filter_;
        transposed_filter.transpose_(0, 1);

        if (InputSize() > BIAS) {
          const auto& bias = Input(BIAS);
          CAFFE_ENFORCE_EQ(bias.ndims(), 1, "bias must be 1D tensor");
          CAFFE_ENFORCE_EQ(
              bias.get_dim(0), filter.get_dim(1),
              "bias dimension must be equal to output channel number");

          ideep::convolution_transpose_forward::compute(
              X, transposed_filter, bias, Y_dims, *Y,
              {stride_.begin(), stride_.end()} , pad_tl(), pad_br());
        } else {
          ideep::convolution_transpose_forward::compute(
              X, transposed_filter, Y_dims, *Y,
              {stride_.begin(), stride_.end()}, pad_tl(), pad_br());
        }
        return true;
        */
    }
}
