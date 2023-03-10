crate::ix!();

pub struct GetSplitGradient;

impl GetGradientDefs for GetSplitGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> output_grads;
        for (int i = 0; i < def_.output_size(); ++i) {
          if (!GradOut(i).IsEmpty()) {
            output_grads.push_back(GO(i));
          }
        }
        if (output_grads.empty()) {
          return {};
        }
        return SingleGradientDef(
            "Concat",
            "",
            output_grads,
            vector<string>{GI(0), "_" + GI(0) + "_dims"});
        */
    }
}

register_gradient!{Split, GetSplitGradient}
register_gradient!{DepthSplit, GetSplitGradient}
register_gradient!{SplitByLengths, GetSplitGradient}

pub struct GetConcatGradient;

impl GetGradientDefs for GetConcatGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (GradOut(0).IsEmpty()) {
          return {};
        }
        vector<string> grads;
        for (int i = 0; i < def_.input_size(); ++i) {
          grads.push_back(GI(i));
        }
        return SingleGradientDef("Split", "", vector<string>{GO(0), O(1)}, grads);
        */
    }
}

register_gradient!{Concat, GetConcatGradient}
register_gradient!{DepthConcat, GetConcatGradient}

register_cuda_operator!{Split, SplitOp<CUDAContext>}
register_cuda_operator!{Concat, ConcatOp<CUDAContext>}
register_cuda_operator!{SplitByLengths, SplitByLengthsOp<CUDAContext>}

