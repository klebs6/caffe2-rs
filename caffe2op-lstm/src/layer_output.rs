crate::ix!();

pub struct LayerOutput<OutputType, HiddenType> {
    outputs:      OutputType,
    final_hidden: HiddenType,
}

impl<OutputType,HiddenType> LayerOutput<OutputType, HiddenType> {
    
    pub fn new(_outputs: &OutputType, _hidden: &HiddenType) -> Self {
        todo!();
        /*
            outputs = copy_ctor(_outputs);
        final_hidden = copy_ctor(_hidden);
        */
    }
}
