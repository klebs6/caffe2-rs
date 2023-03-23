crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPSigmoidOp {
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

