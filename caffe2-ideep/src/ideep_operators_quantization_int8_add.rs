crate::ix!();

///----------------------------------------
pub struct IDEEPInt8SumReluOp<const ReluFused: bool> {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    scale:       f32,
    zero_point:  i32,
    y_scales:    IScale,
    y_data_type: IDType,

}

input_tags!{
    IDEEPInt8SumReluOp {
        Input0
    }
}

output_tags!{
    IDEEPInt8SumReluOp {
        Output
    }
}

impl<const ReluFused: bool> IDEEPInt8SumReluOp<ReluFused> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
            zero_point_( this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) 

        if (ReluFused || zero_point_ == 0) {
          Y_data_type_ = idtype::u8;
          CAFFE_ENFORCE_EQ(zero_point_, 0, "Wrong zero point");
        } else {
          Y_data_type_ = idtype::s8;
          CAFFE_ENFORCE_EQ(zero_point_, 128, "Wrong zero point");
        }

        Y_scales_ = ConvertScales({scale_});
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            itensor temp_ten;
        itensor::dims input_dims;
        vector<itensor> inputs_itensor;

        CAFFE_ENFORCE_GT(InputSize(), 1, "Wrong input size (must > 1)");
        for (int i = 0; i < InputSize(); ++i) {
          CAFFE_ENFORCE(OperatorStorage::InputBlob(i).template IsType<itensor>());
          auto& Xi = Input(i);
          if (input_dims.empty())
            input_dims = Xi.get_dims();
          CAFFE_ENFORCE(input_dims == Xi.get_dims());
          inputs_itensor.emplace_back(
              Xi.get_data_type() != idtype::f32 ? Xi.dequantize() : Xi);
        }

        temp_ten.init({input_dims, idtype::f32});
        const vector<float> scales(InputSize(), 1.0);
        ideep::sum::compute(scales, inputs_itensor, temp_ten);
        if (ReluFused) {
          ideep::eltwise_forward::compute(temp_ten, temp_ten);
        }

        auto* Y = Output(OUTPUT);
        Y->init({temp_ten.get_dims(), Y_data_type_, iformat::nhwc});
        Y->set_scale(Y_scales_);
        Y->feed_from(temp_ten);
        return true;
        */
    }
}

register_ideep_operator_with_engine!{
    Int8Sum, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<false>
}

register_ideep_operator_with_engine!{
    Int8Add, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<false>
}

register_ideep_operator_with_engine!{
    Int8SumRelu, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<true>
}

register_ideep_operator_with_engine!{
    Int8AddRelu, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<true>
}
