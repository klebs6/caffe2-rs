crate::ix!();

use crate::{
    Workspace,
    IDEEPOperator,
    OperatorDef
};

#[inline] pub fn momentum_sgd_update(
    n:         i32,
    g:         *const f32,
    m:         *const f32,
    ng:        *mut f32,
    nm:        *mut f32,
    lr:        *const f32,
    momentum:  f32,
    nesterov:  bool,
    param:     *mut f32)  
{
    todo!();
    /*
        const float LR = lr[0];
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
      for (auto i = 0; i < N; ++i) {
        if (!nesterov) {
          const float adjusted_gradient = LR * g[i] + momentum * m[i];
          nm[i] = adjusted_gradient;
          ng[i] = adjusted_gradient;
        } else {
          const float mi = m[i];
          const float mi_new = momentum * mi + LR * g[i];
          nm[i] = mi_new;
          ng[i] = (1 + momentum) * mi_new - momentum * mi;
        }

        if (param) {
          param[i] -= ng[i];
        }
      }
    */
}

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

pub struct IDEEPMomentumSGDUpdateOp {
    momentum: f32, //0.9
    nesterov: bool,
}

input_tags!{
    IDEEPMomentumSGDUpdateOp {
        Grad, 
        Momentum, 
        Lr, 
        Param
    }
}

output_tags!{
    IDEEPMomentumSGDUpdateOp {
        OutputGrad, 
        OutputMomentum, 
        OutputParam
    }
}

impl IDEEPMomentumSGDUpdateOp {
    
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
            static_cast<float*>(Output(OUTPUT_PARAM)->get_data_handle()));
        return true;
        */
    }
}

register_ideep_operator!{MomentumSGD,       IDEEPMomentumSGDOp}

register_ideep_operator!{MomentumSGDUpdate, IDEEPMomentumSGDUpdateOp}
