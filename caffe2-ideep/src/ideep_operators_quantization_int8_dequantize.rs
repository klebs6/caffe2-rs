crate::ix!();


pub struct IDEEPInt8DequantizeOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    Y_fmt: IFormat, //IFormat::Undef
}

impl IDEEPInt8DequantizeOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws) 

        if (HasArgument("output_order")) {
          Y_fmt_ = static_cast<iformat>(
            this->template GetSingleArgument<int>("output_order",
                                                  static_cast<int>(iformat::nchw)));
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        auto* Y = Output(0);
        if (Y_fmt_ != iformat::undef) {
          Y->init(X.get_desc().to_type(idtype::f32).to_format(Y_fmt_));
        } else {
          Y->init(X.get_desc().to_type(idtype::f32));
        }
        Y->feed_from(X);

        return true;
        */
    }
}

register_ideep_operator_with_engine!{Int8Dequantize, DNNLOWP, IDEEPInt8DequantizeOp}
