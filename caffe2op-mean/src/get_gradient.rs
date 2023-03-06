crate::ix!();

register_cpu_operator!{Mean,         MeanOp<CPUContext>}

register_cpu_operator!{MeanGradient, MeanGradientOp<CPUContext>}

pub struct GetMeanGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMeanGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            auto outputs = std::vector<string>();
        for (int i = 0; i < def_.input_size(); i++) {
          outputs.push_back(GI(i));
        }
        return SingleGradientDef(
            "MeanGradient", "", std::vector<string>{GO(0)}, outputs);
        */
    }
}

register_gradient!{Mean, GetMeanGradient}
