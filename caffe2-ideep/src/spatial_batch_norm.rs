crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPSpatialBNOp {
    base:     IDEEPOperator,
    is_test:  bool,
    epsilon:  f64,
    momentum: f64,
}

input_tags!{
    IDEEPSpatialBNOp {
        Input,
        Scale,
        Bias,
        EstMean,
        EstVar
    }
}

output_tags!{
    IDEEPSpatialBNOp {
        Output,
        RunningMean,
        RunningVar,
        SavedMean,
        SavedVar
    }
}

impl IDEEPSpatialBNOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            is_test_(OperatorStorage::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
            epsilon_(OperatorStorage::GetSingleArgument<float>("epsilon", 1e-5)),
            momentum_(OperatorStorage::GetSingleArgument<float>("momentum", 0.9)) 

        CAFFE_ENFORCE(
            (is_test_ && OutputSize() > OUTPUT)
              || (!is_test_ && OutputSize() > SAVED_VAR));
        CAFFE_ENFORCE_GT(epsilon_, 0);
        CAFFE_ENFORCE_GE(momentum_, 0);
        CAFFE_ENFORCE_LE(momentum_, 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& scale = Input(SCALE);
        const auto& bias = Input(BIAS);
        auto* Y = Output(OUTPUT);

        DCHECK_EQ(scale.ndims(), 1);
        DCHECK_EQ(bias.ndims(), 1);
        DCHECK_EQ(scale.get_dim(0), X.get_dim(1));
        DCHECK_EQ(bias.get_dim(0), X.get_dim(1));

        if (is_test_) {
          const auto& est_mean = Input(EST_MEAN);
          const auto& est_var = Input(EST_VAR);
          auto X_ = X.get_data_type() != idtype::f32 ? X.dequantize() : X;
          ideep::batch_normalization_forward_inference::compute(
              X_, est_mean, est_var, scale, bias, *Y, epsilon_);
        } else {
          auto* saved_mean = Output(SAVED_MEAN);
          auto* saved_var = Output(SAVED_VAR);
          auto* running_mean = Output(RUNNING_MEAN);
          auto* running_var = Output(RUNNING_VAR);
          ideep::batch_normalization_forward_training::compute(
              X, scale, bias, *Y, *saved_mean, *saved_var,
              *running_mean, *running_var, momentum_, epsilon_);
        }

        return true;
        */
    }
}

