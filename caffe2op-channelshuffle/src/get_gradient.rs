crate::ix!();

pub struct GetChannelShuffleGradient;

impl GetGradientDefs for GetChannelShuffleGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ChannelShuffleGradient",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{
    ChannelShuffle, 
    GetChannelShuffleGradient
}
