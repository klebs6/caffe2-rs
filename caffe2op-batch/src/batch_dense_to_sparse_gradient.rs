crate::ix!();

///---------------------------------------------------------

pub struct GetBatchDenseToSparseGradient;

impl GetGradientDefs for GetBatchDenseToSparseGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchSparseToDense",
            "",
            vector<string>{I(0), I(1), GO(0), I(2)},
            vector<string>{GI(2)});
        */
    }
}


register_gradient!{BatchDenseToSparse, GetBatchDenseToSparseGradient}

