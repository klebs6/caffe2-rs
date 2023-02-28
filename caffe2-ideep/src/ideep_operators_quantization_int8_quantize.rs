crate::ix!();

use crate::{
    IFormat,
    IDType,
    OperatorDef,
    IScale,
    IDEEPOperator,
    Workspace
};

pub struct IDEEPInt8QuantizeOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    scale:       f32,
    zero_point:  i32,
    y_scales:    IScale,
    y_data_type: IDType,
    y_fmt:       IFormat, // {iformat::undef};
}

impl IDEEPInt8QuantizeOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
            zero_point_(
                this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) 

        if (HasArgument("output_order")) {
          Y_fmt_ = static_cast<iformat>(
            this->template GetSingleArgument<int>("output_order",
                                                  static_cast<int>(iformat::nchw)));
        }

        CAFFE_ENFORCE(zero_point_ == 0 || zero_point_ == 128,
            "Not support this zero point");
        Y_data_type_ = zero_point_ == 0 ? idtype::u8 : idtype::s8;
        Y_scales_ = ConvertScales({scale_});
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        CAFFE_ENFORCE(X.get_data_type() == idtype::f32, "Not support data type");

        auto* Y = Output(0);
        if (Y_fmt_ != iformat::undef) {
          Y->init(X.get_desc().to_type(Y_data_type_).to_format(Y_fmt_));
        } else {
          Y->init(X.get_desc().to_type(Y_data_type_));
        }
        Y->set_scale(Y_scales_);
        Y->feed_from(X);

        return true;
        */
    }
}

input_tags!{
    IDEEPInt8QuantizeOp {
        Input0
    }
}

output_tags!{
    IDEEPInt8QuantizeOp {
        Output
    }
}

register_ideep_operator_with_engine!{
    Int8Quantize, 
    DNNLOWP, 
    IDEEPInt8QuantizeOp
}
