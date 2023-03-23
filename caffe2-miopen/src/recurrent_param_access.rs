crate::ix!();

#[USE_RECURRENT_BASE_FUNCTIONS]
pub struct RecurrentParamAccessOp<T,const mode: RecurrentParamOpMode> {
    base:    RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

impl<T,const mode: RecurrentParamOpMode> RecurrentParamAccessOp<T,mode> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(operator_def, ws)
        */
    }
}
