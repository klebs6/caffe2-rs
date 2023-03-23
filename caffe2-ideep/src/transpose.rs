crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPTransposeOp {
    base: IDEEPOperator,
    axes: Vec<i32>,
} 

input_tags!{
    IDEEPTransposeOp {
        Input
    }
}

output_tags!{
    IDEEPTransposeOp {
        Output
    }
}

impl IDEEPTransposeOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            axes_(this->template GetRepeatedArgument<int>("axes"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);

        Y->transpose_from(X.to_public(nullptr, X.get_data_type()), axes_);

        return true;
        */
    }
}
