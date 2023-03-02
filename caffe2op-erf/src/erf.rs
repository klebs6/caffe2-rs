crate::ix!();

/**
  | Calculates the arcsine of the given
  | input tensor, element-wise.
  |
  */
pub struct Erf {

}

num_inputs!{Erf, 1}

num_outputs!{Erf, 1}

inputs!{Erf, 
    0 => ("input", "Input tensor")
}

outputs!{Erf, 
    0 => ("output", "The arcsine of the input tensor computed element-wise")
}

identical_type_and_shape!{Erf}

pub struct ErfGradient {

}

num_inputs!{ErfGradient, 2}

num_outputs!{ErfGradient, 1}

identical_type_and_shape!{ErfGradient}

pub struct GetErfGradient;

impl GetGradientDefs for GetErfGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ErfGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Erf, GetErfGradient}
