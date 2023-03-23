crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
pub struct IDEEPCopyOp {
    base: IDEEPOperator,
} 

impl IDEEPCopyOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = OperatorStorage::Input<itensor>(0);
        auto* Y = Output(0);
        if (X != *Y) {
          Y->reinit_like(X);
          ideep::direct_copy::compute(X, *Y);
        }

        return true;
        */
    }
}

