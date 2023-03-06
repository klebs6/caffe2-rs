crate::ix!();

pub struct FullBidirectionalLSTMLayer {
    layer:   FullLSTMLayer,
    context: *mut CPUContext,
}

impl Layer<(TensorTuple, TensorTuple), (CellParams, CellParams)> for FullBidirectionalLSTMLayer {

    #[inline] fn invoke(
        &self, 
        input: &Tensor,
        input_hidden: &<Self as FullBidirectionalLSTMLayerTypes>::BidirHiddenType,
        params: &<Self as FullBidirectionalLSTMLayerTypes>::ParamType) 
        -> <Self as FullBidirectionalLSTMLayerTypes>::OutputType 
    {

        todo!();
        /*
            std::vector<Tensor> outputs;
        auto step_inputs = unbind(input, 0, context_);
        auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
        auto fw_output = stack(fw_result.outputs, 0, context_);
        outputs.push_back(copy_ctor(fw_output));
        auto rev_step_inputs = reverse(std::move(step_inputs));
        auto rev_result =
            layer_(rev_step_inputs, input_hidden.second, params.second);
        std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
        auto rev_output = stack(rev_result.outputs, 0, context_);
        outputs.push_back(copy_ctor(rev_output));
        return {cat(outputs, fw_output.dim() - 1, context_),
                std::make_pair(
                    std::move(fw_result.final_hidden),
                    std::move(rev_result.final_hidden))};
        */
    }
}

impl FullBidirectionalLSTMLayerTypes for FullBidirectionalLSTMLayer { }

impl FullBidirectionalLSTMLayer {
    
    pub fn new(cell: &mut LSTMCell, context: *mut CPUContext) -> Self {
        todo!();
        /*
            : layer_(cell, context), context_(context)
        */
    }
    
    #[inline] pub fn reverse(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        
        todo!();
        /*
            std::reverse(x.begin(), x.end());
        return std::move(x);
        */
    }
}
