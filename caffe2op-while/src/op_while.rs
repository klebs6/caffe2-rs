crate::ix!();

use crate::{
    OperatorStorage,
    Workspace,
    NetDef,
    OperatorDef,
    NetBase
};

/**
  | 'While' control operator, first input
  | is a scalar boolean blob that stores
  | loop's condition value.
  | 
  | Accepts 'loop_net' (required) and
  | 'cond_net' (optional) arguments for
  | loop's body and condition subnets respectively.
  | 
  | If condition subnet is specified, it
  | is executed before the first and after
  | each iteration.
  | 
  | Subnets are executed in the same workspace
  | as 'While'.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WhileOp<Context> {

    storage:      OperatorStorage,
    context:      Context,

    loop_net_def: NetDef,
    loop_net:     Box<NetBase>,
    cond_net_def: NetDef,
    cond_net:     Box<NetBase>,
}

num_inputs!{While, (1,INT_MAX)}

num_outputs!{While, (0,INT_MAX)}

inputs!{While, 
    0 => ("condition", "Scalar boolean condition")
}

args!{While, 
    0 => ("loop_net", "Net executed on each iteration"),
    1 => ("cond_net", "Net to (re)compute condition value")
}

allow_inplace!{While, 
    |input: i32, output: i32| -> bool {
        true
    }
}

register_cpu_operator!{While,  WhileOp<CPUContext>}

register_cuda_operator!{While, WhileOp<CUDAContext>}

impl<Context> WhileOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws) 
        CAFFE_ENFORCE(
            this->template HasSingleArgumentOfType<NetDef>("loop_net"),
            "loop_net must be specified in While operator");
        loop_net_def_ =
            this->template GetSingleArgument<NetDef>("loop_net", NetDef());
        loop_net_ = CreateNet(loop_net_def_, ws);
        CAFFE_ENFORCE(loop_net_, "Failed to initialize loop subnet");

        cond_net_ = nullptr;
        bool has_cond_net =
            this->template HasSingleArgumentOfType<NetDef>("cond_net");
        if (has_cond_net) {
          cond_net_def_ =
              this->template GetSingleArgument<NetDef>("cond_net", NetDef());
          cond_net_ = CreateNet(cond_net_def_, ws);
          CAFFE_ENFORCE(cond_net_, "Failed to initialize condition subnet");
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            this->InputIsTensorType(0, Context::GetDeviceType()),
            "Invalid condition in While operator: tensor expected");

        const auto& condition = Input(0);
        CAFFE_ENFORCE_EQ(
            condition.numel(),
            1,
            "Invalid condition tensor in While operator: single value expected");

        while (true) {
          if (cond_net_ && !cond_net_->Run()) {
            return false;
          }
          if (!*condition.template data<bool>()) {
            return true;
          }
          if (!loop_net_->Run()) {
            return false;
          }
        }

        return true;
        */
    }
}
