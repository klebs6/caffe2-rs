crate::ix!();

use crate::{
    IDEEPOperator,
    Workspace,
    OperatorDef
};

pub struct IDEEPInt8ReluOp {

    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base:  IDEEPOperator,
    alpha: f32,

}

input_tags!{
    IDEEPInt8ReluOp {
        Input
    }
}

output_tags!{
    IDEEPInt8ReluOp {
        Output
    }
}

impl IDEEPInt8ReluOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws), alpha_(0.0) 

        // Figure out the Relu descriptor.
        if (operator_def.type().substr(0, 8) == "Int8Relu") {
          alpha_ = 0.0;
        } else {
          LOG(FATAL) << "Unsupported Relu method: " << operator_def.type();
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);

        ideep::eltwise_forward::compute(
            X, *Y, ialgo::eltwise_relu, iprop::forward_inference, alpha_);

        return true;
        */
    }
}

register_ideep_operator_with_engine!{Int8Relu, DNNLOWP, IDEEPInt8ReluOp}
