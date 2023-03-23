crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPReluGradientOp {
    base:  IDEEPOperator,
    alpha: f32,
}

input_tags!{
    IDEEPReluGradientOp {
        Output,
        OutputGrad
    }
}
output_tags!{
    IDEEPReluGradientOp {
        InputGrad
    }
}

impl IDEEPReluGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws), alpha_(0.0) 

        // Figure out the Relu descriptor.
        if (operator_def.type().substr(0, 12) == "ReluGradient") {
          alpha_ = 0.0;
        } else if (operator_def.type().substr(0, 17) == "LeakyReluGradient") {
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
            const auto& Y = Input(OUTPUT);
        const auto& dY = Input(OUTPUT_GRAD);
        auto* dX = Output(INPUT_GRAD);

        ideep::eltwise_backward::compute(Y, dY, *dX, ialgo::eltwise_relu, alpha_);

        return true;
        */
    }
}
