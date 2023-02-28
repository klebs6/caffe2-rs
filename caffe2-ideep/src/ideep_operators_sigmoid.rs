crate::ix!();

use crate::{
    Workspace,
    OperatorDef,
    IDEEPOperator
};

pub struct IDEEPSigmoidOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,
}

input_tags!{
    IDEEPSigmoidOp {
        Input
    }
}

output_tags!{
    IDEEPSigmoidOp {
        Output
    }
}

impl IDEEPSigmoidOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);

        ideep::eltwise_forward::compute(
            X, *Y, ialgo::eltwise_logistic, iprop::forward_training);

        return true;
        */
    }
}

///------------------------
pub struct IDEEPSigmoidGradientOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,
}

input_tags!{
    IDEEPSigmoidGradientOp {
        Output,
        OutputGrad
    }
}

output_tags!{
    IDEEPSigmoidGradientOp {
        InputGrad
    }
}

impl IDEEPSigmoidGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(OUTPUT);
        const auto& dY = Input(OUTPUT_GRAD);
        auto* dX = Output(INPUT_GRAD);

        ideep::eltwise_backward::compute(Y, dY, *dX, ialgo::eltwise_logistic);

        return true;
        */
    }
}

register_ideep_operator!{Sigmoid,         IDEEPSigmoidOp}

register_ideep_operator!{SigmoidGradient, IDEEPSigmoidGradientOp}
