crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPSpatialBNGradientOp {
    base:    IDEEPOperator,
    epsilon: f64,
} 

input_tags!{
    IDEEPSpatialBNGradientOp {
        Input,
        Scale,
        OutputGrad,
        SavedMean,
        SavedVar
    }
}

output_tags!{
    IDEEPSpatialBNGradientOp {
        InputGrad,
        ScaleGrad,
        BiasGrad
    }
}

impl IDEEPSpatialBNGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            epsilon_(OperatorStorage::GetSingleArgument<float>("epsilon", 1e-5)) 

        CAFFE_ENFORCE(InputSize() > SAVED_VAR);
        CAFFE_ENFORCE(OutputSize() > BIAS_GRAD);
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& scale = Input(SCALE);
        const auto& dY = Input(OUTPUT_GRAD);
        const auto& saved_mean = Input(SAVED_MEAN);
        const auto& saved_var = Input(SAVED_VAR);
        auto* dX = Output(INPUT_GRAD);
        auto* dscale = Output(SCALE_GRAD);
        auto* dbias = Output(BIAS_GRAD);

        ideep::batch_normalization_backward::compute(
            X, saved_mean, saved_var, dY, scale,
            *dX, *dscale, *dbias, epsilon_);

        return true;
        */
    }
}
