crate::ix!();

use crate::{
    OperatorStorage,
    Workspace,
    Operator,
    OperatorDef
};

/**
 | Computes a momentum SGD update for an input
 | gradient and momentum parameters. Concretely,
 | given inputs (grad, m, lr) and parameters
 | (momentum, nesterov), computes:
 |
 |     if not nesterov:
 |         adjusted_gradient = lr * grad + momentum * m
 |         return (adjusted_gradient, adjusted_gradient)
 |     else:
 |         m_new = momentum * m + lr * grad
 |         return ((1 + momentum) * m_new - momentum * m, m_new)
 |
 | Output is (grad, momentum)
 |
 | Note the difference to MomemtumSGDUpdate, which
 | actually performs the parameter update (and is
 | thus faster).
 */
pub struct MomentumSGDOp<T, Context> {
    context:  Context,
    momentum: T, // default = 0.9
    nesterov: bool,
}

impl<T,Context> Operator for MomentumSGDOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;

}

register_cpu_operator!{MomentumSGD, MomentumSGDOp<f32, CPUContext>}

num_inputs!{MomentumSGD, 3}

num_outputs!{MomentumSGD, 2}

tensor_inference_function!{MomentumSGD, /* [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(2);
          out[0] = in[0];
          out[1] = in[1];
          return out;
        } */}

allow_inplace!{MomentumSGD, vec![(0, 0), (1, 1)]}

should_not_do_gradient!{MomentumSGD}

input_tags!{
    MomentumSGDOp
    {
        Grad,
        Momentum,
        Lr
    }
}

output_tags!{
    MomentumSGDOp
    {
        OutputGrad,
        OutputMomentum
    }
}

impl<T,Context> MomentumSGDOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
            nesterov_(this->template GetSingleArgument<bool>("nesterov", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto device_type = Context::GetDeviceType();
        // Iter live on the CPU
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(GRAD, device_type));
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(MOMENTUM, device_type));
        CAFFE_ENFORCE(Input(LR).numel() == 1);
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENTUM).numel());
        Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
        Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

        momentum_sgd_update<Context>(
            Input(GRAD).numel(),
            Input(GRAD).template data<T>(),
            Input(MOMENTUM).template data<T>(),
            Output(OUTPUT_GRAD)->template mutable_data<T>(),
            Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
            Input(LR).template data<T>(),
            momentum_,
            nesterov_,
            NULL,
            &context_);
        return true;
        */
    }
}

/**
 | Performs a momentum SGD update for an input
 | gradient and momentum parameters. Concretely,
 | given inputs (grad, m, lr, param) and arguments
 | (momentum, nesterov), computes:
 |
 |     if not nesterov:
 |         adjusted_gradient = lr * grad + momentum * m
 |         param = param - adjusted_gradient
 |         return (adjusted_gradient, adjusted_gradient, param)
 |     else:
 |         m_new = momentum * m + lr * grad
 |         param = param - ((1 + momentum) * m_new - momentum * m),
 |         return ((1 + momentum) * m_new - momentum * m, m_new, param)
 |
 | Output is (grad, momentum, parameter).
 |
 | Note the difference to MomentumSGD, which returns
 | a new gradient but does not perform the parameter
 | update.
 */
pub struct MomentumSGDUpdateOp<T, Context> {
    context:  Context,
    momentum: T, // default = 0.9
    nesterov: bool,
}

impl<T,Context> Operator for MomentumSGDUpdateOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;

}

register_cpu_operator!{
    MomentumSGDUpdate,
    MomentumSGDUpdateOp<f32, CPUContext>
}

num_inputs!{MomentumSGDUpdate, 4}

num_outputs!{MomentumSGDUpdate, 3}

allow_inplace!{MomentumSGDUpdate, vec![(0, 0), (1, 1), (3, 2)]}

tensor_inference_function!{MomentumSGDUpdate, /*
    [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
      vector<TensorShape> out(3);
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[3];
      return out;
    } */
}

should_not_do_gradient!{MomentumSGDUpdate}

input_tags!{
    MomentumSGDUpdateOp
    {
        Grad,
        Momentum,
        Lr,
        Param
    }
}

output_tags!{
    MomentumSGDUpdateOp
    {
        OutputGrad,
        OutputMomentum,
        OutputParam
    }
}

impl<T,Context> MomentumSGDUpdateOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
            nesterov_(this->template GetSingleArgument<bool>("nesterov", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto device_type = Context::GetDeviceType();
        // Iter live on the CPU
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(GRAD, device_type));
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(MOMENTUM, device_type));
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(MOMENTUM).numel());
        Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
        Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

        momentum_sgd_update<Context>(
            Input(GRAD).numel(),
            Input(GRAD).template data<T>(),
            Input(MOMENTUM).template data<T>(),
            Output(OUTPUT_GRAD)->template mutable_data<T>(),
            Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
            Input(LR).template data<T>(),
            momentum_,
            nesterov_,
            Output(OUTPUT_PARAM)->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

/**
  | Performs a momentum SGD update analogous
  | to MomentumSGDUpdate, but using a GradientSlice
  | and indices into the full param and momentum
  | tables. Both param and momentum should
  | be in-place (corresponding inputs
  | and outputs should be the same blobs).
  |
  */
pub struct SparseMomentumSGDUpdateOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    momentum: T,
    nesterov: bool,
}

input_tags!{
    SparseMomentumSGDOp
    {
        Grad,
        Momentum,
        Lr,
        Param,
        Indices
    }
}

output_tags!{
    SparseMomentumSGDOp
    {
        OutputGrad,
        OutputMomentum,
        OutputParam
    }
}

register_cpu_operator!{
    SparseMomentumSGDUpdate,
    SparseMomentumSGDUpdateOp<f32, CPUContext>
}

num_inputs!{SparseMomentumSGDUpdate, 5}

num_outputs!{SparseMomentumSGDUpdate, 3}

inputs!{SparseMomentumSGDUpdate, 
    0 => ("grad", "GradientSlice with gradients for updated indices."),
    1 => ("moment", "Momentum blob, same shape as param."),
    2 => ("lr", "Learning rate."),
    3 => ("param", "Full parameter blob."),
    4 => ("indices", "Indices (in first dimension of param) where updates are performed.")
}

outputs!{SparseMomentumSGDUpdate, 
    0 => ("output_grad", "Adjusted gradient."),
    1 => ("output_moment", "Updated momentum."),
    2 => ("output_param", "Updated parameter")
}

args!{SparseMomentumSGDUpdate, 
    0 => ("momentum", "Momentum hyperparameter."),
    1 => ("nesterov", "(boolean) Whether to use Nesterov Accelerated Gradient.")
}

tensor_inference_function!{SparseMomentumSGDUpdate, /* [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(3);
          out[0] = in[0];
          out[1] = in[1];
          out[2] = in[3];
          return out;
        } */
}

allow_inplace!{SparseMomentumSGDUpdate, vec![(0, 0)]}

enforce_inplace!{SparseMomentumSGDUpdate, vec![(1, 1), (3, 2)]}

should_not_do_gradient!{SparseMomentumSGDUpdate}

impl<T,Context> SparseMomentumSGDUpdateOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
            nesterov_(this->template GetSingleArgument<bool>("nesterov", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Resize [potentially] out-of-place blobs
        Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));

        // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENTUM).numel());
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self, ) -> bool {
        todo!();
        /*
            auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
        auto n = Input(GRAD).numel() / block_size;

        const auto* gradIn = Input(GRAD).template data<T>();
        const auto* momentumIn = Input(MOMENTUM).template data<T>();
        const auto* lr = Input(LR).template data<T>();
        // const auto* paramIn = Input(PARAM).template data<T>();
        const auto* indices = Input(INDICES).template data<SIndex>();

        auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
        auto* momentumOut = Output(OUTPUT_MOMENTUM)->template mutable_data<T>();
        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();

        for (auto i = 0; i < n; ++i) {
          auto idx = indices[i];
          auto offsetI = i * block_size;
          auto offsetIdx = idx * block_size;

          CAFFE_ENFORCE(offsetIdx + block_size <= Input(PARAM).numel());
          CAFFE_ENFORCE(offsetI + block_size <= Input(GRAD).numel());

          momentum_sgd_update<Context>(
              block_size,
              gradIn + offsetI,
              momentumIn + offsetIdx,
              gradOut + offsetI,
              momentumOut + offsetIdx,
              lr,
              momentum_,
              nesterov_,
              paramOut + offsetIdx,
              &context_);
        }
        return true;
        */
    }
}

///------------------------------------------------
#[inline] pub fn momentum_sgd_update<Context>(
    n:          i32,
    g:          *const f32,
    m:          *const f32,
    ng:         *mut f32,
    nm:         *mut f32,
    lr:         *const f32,
    momentum:   f32,
    nesterov:   bool,
    param:      *mut f32,
    context:    *mut Context) 
{
    todo!();
    /*
        const float LR = lr[0];
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

