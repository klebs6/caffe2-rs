crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPReluOp {
    base: IDEEPOperator,
    alpha: f32,
}

input_tags!{
    IDEEPReluOp {
        Input
    }
}

output_tags!{
    IDEEPReluOp {
        Output
    }
}

impl IDEEPReluOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws), alpha_(0.0) 

        // Figure out the Relu descriptor.
        if (operator_def.type().substr(0, 4) == "Relu") {
          alpha_ = 0.0;
        } else if (operator_def.type().substr(0, 9) == "LeakyRelu") {
          if (HasArgument("alpha")) {
            alpha_ = static_cast<float>(
                OperatorStorage::GetSingleArgument<float>("alpha", 0.01));
          }
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
            X, *Y, ialgo::eltwise_relu, iprop::forward_training, alpha_);

        return true;
        */
    }
}
