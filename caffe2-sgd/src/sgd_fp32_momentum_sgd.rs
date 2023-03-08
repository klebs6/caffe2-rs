crate::ix!();

use crate::{
    Operator,
    Workspace,
    OperatorDef
};

#[inline] pub fn fp32_momentum_sgd_update<Context>(
    n:            i32,
    g:            *const f32,
    m:            *const f32,
    ng:           *mut f32,
    nm:           *mut f32,
    lr:           *const f32,
    momentum:     f32,
    nesterov:     bool,
    weight_decay: f32,
    param:        *mut f32,
    context:      *mut Context)  {

    todo!();
    /*
    
    */
}

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FP32MomentumSGDUpdateOp<T,Context> {
    context: Context,
    momentum:      f32, // default = 0.9
    weight_decay:  f32, // default = 0.0
    nesterov:      bool,
    phantom: PhantomData<T>,
}

impl<T, Context> Operator for FP32MomentumSGDUpdateOp<T,Context> {
}

input_tags!{
    FP32MomentumSGDUpdateOp
    {
        Grad,
        Momentum,
        Lr,
        Param
    }
}

output_tags!{
    FP32MomentumSGDUpdateOp
    {
        OutputGrad,
        OutputMomentum,
        OutputParam
    }
}

impl<T,Context> FP32MomentumSGDUpdateOp<T,Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            momentum_(this->template GetSingleArgument<float>("momentum", 0.0)),
            weight_decay_(
                this->template GetSingleArgument<float>("weight_decay", 0.0)),
            nesterov_(this->template GetSingleArgument<int>("nesterov", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto device_type = Context::GetDeviceType();
        // Iter live on the CPU
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(GRAD, device_type));
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(MOMENTUM, device_type));
        CAFFE_ENFORCE(Input(LR).size() == 1);
        CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENTUM).size());
        Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
        Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

        fp32_momentum_sgd_update<Context>(
            Input(GRAD).size(),
            Input(GRAD).template data<T>(),
            Input(MOMENTUM).template data<T>(),
            Output(OUTPUT_GRAD)->template mutable_data<T>(),
            Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
            Input(LR).template data<float>(),
            momentum_,
            nesterov_,
            weight_decay_,
            Output(OUTPUT_PARAM)->template mutable_data<T>(),
            &context_);

        return true;
        */
    }
}
