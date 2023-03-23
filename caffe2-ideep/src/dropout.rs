crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPDropoutOp {
    base:    IDEEPOperator,
    ratio:   f32,
    is_test: bool,
}

input_tags!{
    IDEEPDropoutOp {
        Input
    }
}

output_tags!{
    IDEEPDropoutOp {
        Output,
        Mask
    }
}

impl IDEEPDropoutOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            ratio_(OperatorStorage::GetSingleArgument<float>("ratio", 0.5)),
            is_test_( OperatorStorage::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) 

        CAFFE_ENFORCE_GE(ratio_, 0);
        CAFFE_ENFORCE_LT(ratio_, 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);

        if (is_test_) {
          if (Y != &X) {
            ideep::direct_copy::compute(X, *Y);
          }
          return true;
        }

        auto* mask = Output(MASK);
        ideep::dropout_forward::compute(X, ratio_, *Y, *mask);

        return true;
        */
    }
}

