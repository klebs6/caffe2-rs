crate::ix!();

pub struct GetRecurrentGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRecurrentGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RecurrentGradient",
            "",
            vector<string>{I(0), // INPUT
                           I(1), // HIDDEN_INPUT
                           I(2), // CELL_INPUT
                           I(3), // WEIGHT
                           O(3), // RNN_SCRATCH
                           O(0), // OUTPUT
                           GO(0)}, // GRAD_OUTPUT
            // TODO: not currently using these gradients, investigate t16675365
            //     GO(1), // GRAD_HIDDEN_OUTPUT
            //     GO(2)}, // GRAD_CELL_OUTPUT
            vector<string>{
                GI(0), // GRAD_INPUT
                GI(1), // GRAD_HIDDEN_INPUT
                GI(2), // GRAD_CELL_INPUT
                GI(3), // GRAD_WEIGHT
                O(4), // DROPOUT_STATES
                O(3) // RNN_SCRATCH
            });
        */
    }
}

register_gradient!{Recurrent, GetRecurrentGradient}
