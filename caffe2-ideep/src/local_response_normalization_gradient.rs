crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPLRNGradientOp {
    base:    IDEEPOperator,
    size:    i32,
    alpha:   f32,
    beta:    f32,
    bias:    f32,
    pre_pad: i32,
}

input_tags!{
    IDEEPLRNGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    IDEEPLRNGradientOp {
        InputGrad
    }
}

impl IDEEPLRNGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            size_(OperatorStorage::GetSingleArgument<int>("size", 0)),
            alpha_(OperatorStorage::GetSingleArgument<float>("alpha", 0)),
            beta_(OperatorStorage::GetSingleArgument<float>("beta", 0)),
            bias_(OperatorStorage::GetSingleArgument<float>("bias", 1)),
            pre_pad_((size_ - 1) / 2) 

        DCHECK_GT(size_, 0);
        DCHECK_EQ(size_ % 2, 1);
        DCHECK_GT(alpha_, 0);
        DCHECK_GT(beta_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& Y = Input(FILTER);
        const auto& dY = Input(OUTPUT_GRAD);
        auto* dX = Output(INPUT_GRAD);

        ideep::lrn_backward::compute(X, dY, Y, *dX, size_, alpha_, beta_, bias_);

        return true;
        */
    }
}
