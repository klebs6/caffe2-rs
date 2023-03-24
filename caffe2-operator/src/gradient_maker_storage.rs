crate::ix!();

pub struct GradientMakerStorage<'a> {
    def:      &'a OperatorDef,
    g_output: &'a Vec<GradientWrapper>,
    g_input:  Vec<GradientWrapper>,
}

impl<'a> GradientMakerStorage<'a> {
    
    pub fn new(
        def:      &OperatorDef, 
        g_output: &Vec<GradientWrapper>) -> Self {

        todo!();
        /*
            : def_(def), g_output_(g_output), g_input_(def.input_size())
        */
    }
}


