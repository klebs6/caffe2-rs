crate::ix!();

pub struct GetSqrGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSqrGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            Argument scale_arg;
        scale_arg.set_name("scale");
        scale_arg.set_f(2.0);
        return std::vector<OperatorDef>{CreateOperatorDef(
                                            "Scale",
                                            "",
                                            std::vector<std::string>{GO(0)},
                                            std::vector<std::string>{GO(0)},
                                            std::vector<Argument>{scale_arg}),
                                        CreateOperatorDef(
                                            "Mul",
                                            "",
                                            std::vector<std::string>{GO(0), I(0)},
                                            std::vector<std::string>{GI(0)})};
        */
    }
}

register_gradient!{Sqr, GetSqrGradient}
