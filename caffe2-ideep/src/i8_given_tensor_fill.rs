crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPInt8GivenTensorFillOp {
    base:        IDEEPOperator,
    scales:      IScale,
    zero_point:  i32,
    fmt:         IFormat,
    shape:       ITensorDims,
    values:      Tensor, //{CPU};
}

output_tags!{
    IDEEPInt8GivenTensorFillOp {
        Output
    }
}

impl IDEEPInt8GivenTensorFillOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            zero_point_(
                this->template GetSingleArgument<int32_t>("Y_zero_point", 0)),
            shape_(this->template GetRepeatedArgument<itensor::dim>("shape")) 

        CAFFE_ENFORCE(shape_.size() == 4 || shape_.size() == 2 || shape_.size() == 1);
        CAFFE_ENFORCE(zero_point_ == 0 || zero_point_ == 128,
            "Not support zero point");
        if (HasArgument("Y_scales")) {
          scales_ = this->template GetRepeatedArgument<float>("Y_scales");
        } else {
          auto scale = (this->template GetSingleArgument<float>("Y_scale", 1.0));
          scales_ = {scale};
        }

        if (shape_.size() == 4) {
          fmt_ = iformat::nhwc;
          auto C = shape_[3];
          shape_[3] = shape_[2];
          shape_[2] = shape_[1];
          shape_[1] = C;
        } else if (shape_.size() == 2) {
          fmt_ = iformat::nc;
        } else {
          fmt_ = iformat::x;
        }

        auto source_values = this->template GetSingleArgument<string>("values", "");
        auto src_size = source_values.size();
        values_.Resize(src_size);
        uint8_t* values_data = values_.template mutable_data<uint8_t>();
        for (int i = 0; i < src_size; i++) {
          values_data[i] = static_cast<uint8_t>(source_values[i]);
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(OUTPUT);
        auto data_type = zero_point_ == 0 ? idtype::u8 : idtype::s8;

        output->init({shape_, data_type});
        DCHECK_EQ(output->get_nelems(), values_.numel())
            << "output size: " << output->get_nelems()
            << " given size: " << values_.numel();

        if (output->get_nelems() > 0) {
          itensor temp_ten;
          temp_ten.init({shape_, data_type, fmt_});
          auto* data_u8 = static_cast<uint8_t*>(temp_ten.get_data_handle());
          const auto* values_data = values_.template data<uint8_t>();
          context_.template CopySameDevice<uint8_t>(
              temp_ten.get_nelems(), values_data, data_u8);

          // Shift quantized data to s8 per zero point
          if (zero_point_ == 128) {
            auto* data_s8 = static_cast<int8_t*>(temp_ten.get_data_handle());
            auto nelems = temp_ten.get_nelems();
            for (int i = 0; i < nelems; i++) {
              data_s8[i] = data_s8[i] - zero_point_;
            }
          }

          output->feed_from(temp_ten);
        }

        output->set_scale(ConvertScales(scales_));
        return true;
        */
    }
}
