crate::ix!();

pub struct FullLSTMLayer {
    cell:    LSTMCell,
    context: *mut CPUContext,
}

impl Layer<TensorTuple, CellParams> for FullLSTMLayer {

    fn invoke(&self,
        inputs: &Tensor,
        input_hidden: &(Tensor, Tensor),
        params: &CellParams) 
        -> LayerOutput<Tensor, TensorTuple> 
    {
        todo!();
        /*
        auto unstacked_output =
            (*this)(unbind(inputs, 0, context_), input_hidden, params);
        return {stack(unstacked_output.outputs, 0, context_),
                unstacked_output.final_hidden};
        */
    }
}

impl FullLSTMLayer {
    
    pub fn new(cell: &mut LSTMCell, context: *mut CPUContext) -> Self {
        todo!();
        /*
            : cell_(cell), context_(context)
        */
    }

    pub fn invoke_vec(&self, 
        step_inputs:  &Vec<Tensor>, 
        input_hidden: (Tensor, Tensor), 
        params:       &CellParams) 
        -> LayerOutput<Vec<Tensor>, TensorTuple> 
    {
        todo!();
        /*
        std::vector<Tensor> step_outputs;
        auto hidden = copy_ctor(input_hidden);

        for (size_t i = 0; i < step_inputs.size(); i++) {
          hidden = cell_(step_inputs[i], hidden, params);
          step_outputs.push_back(copy_ctor(std::get<0>(hidden)));
        }

        return {step_outputs, hidden};
        */
    }
}
