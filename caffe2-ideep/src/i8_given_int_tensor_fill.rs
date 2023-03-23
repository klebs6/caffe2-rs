crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPInt8GivenIntTensorFillOp {
    base:       IDEEPOperator,
    scales:     IScale,
    zero_point: i32,
    shape:      ITensorDims,
    values:     Tensor, //{CPU};
}

output_tags!{
    IDEEPInt8GivenIntTensorFillOp {
        Output
    }
}

impl IDEEPInt8GivenIntTensorFillOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            zero_point_(
                this->template GetSingleArgument<int32_t>("Y_zero_point", 0)),
            shape_(this->template GetRepeatedArgument<itensor::dim>("shape")) 

        CAFFE_ENFORCE(zero_point_ == 0, "Not support zero point");
        if (HasArgument("Y_scales")) {
          scales_ = this->template GetRepeatedArgument<float>("Y_scales");
        } else {
          auto scale = (this->template GetSingleArgument<float>("Y_scale", 1.0));
          scales_ = {scale};
        }

        auto source_values = this->template GetRepeatedArgument<int32_t>("values");
        auto src_size = source_values.size();
        values_.Resize(src_size);
        auto* values_data = values_.template mutable_data<int32_t>();
        for (int i = 0; i < src_size; i++) {
          values_data[i] = static_cast<int32_t>(source_values[i]);
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(OUTPUT);
        output->init({shape_, idtype::s32});
        output->set_scale(ConvertScales(scales_));
        DCHECK_EQ(output->get_nelems(), values_.numel())
            << "output size: " << output->get_nelems()
            << " given size: " << values_.numel();

        if (output->get_nelems() > 0) {
          auto* data = static_cast<int32_t*>(output->get_data_handle());
          const int32_t* values_data = values_.template data<int32_t>();
          context_.template CopySameDevice<int32_t>(
              output->get_nelems(), values_data, data);
        }
        return true;
        */
    }
}

