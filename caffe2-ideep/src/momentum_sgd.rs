crate::ix!();

pub struct IDEEPMomentumSGDOp {
    momentum: f32,//0.9
    nesterov: bool,
}

input_tags!{
    IDEEPMomentumSGDOp {
        Grad, 
        Momentum, 
        Lr
    }
}

output_tags!{
    IDEEPMomentumSGDOp {
        OutputGrad, 
        OutputMomentum
    }
}

impl IDEEPMomentumSGDOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            momentum_(OperatorStorage::GetSingleArgument<float>("momentum", 0.0)),
            nesterov_(OperatorStorage::GetSingleArgument<int>("nesterov", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(Input(GRAD).get_nelems() == Input(MOMENTUM).get_nelems());
        if (Input(GRAD) != *Output(OUTPUT_GRAD)) {
          Output(OUTPUT_GRAD)->init(Input(GRAD).get_descriptor());
        }
        if (Input(MOMENTUM) != *Output(OUTPUT_MOMENTUM)) {
          Output(OUTPUT_MOMENTUM)->init(Input(MOMENTUM).get_descriptor());
        }

        // TODO: Use itensor after 0-dim is supported. Now use CPU tensor.
        const auto& lr = OperatorStorage::Input<TensorCPU>(LR, CPU);
        CAFFE_ENFORCE(lr.numel() == 1);

        momentum_sgd_update(
            Input(GRAD).get_nelems(),
            static_cast<float*>(Input(GRAD).get_data_handle()),
            static_cast<float*>(Input(MOMENTUM).get_data_handle()),
            static_cast<float*>(Output(OUTPUT_GRAD)->get_data_handle()),
            static_cast<float*>(Output(OUTPUT_MOMENTUM)->get_data_handle()),
            lr.template data<float>(),
            momentum_,
            nesterov_,
            nullptr);
        return true;
        */
    }
}
