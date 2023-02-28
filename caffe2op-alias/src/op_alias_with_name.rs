crate::ix!();

use crate::{
    OperatorStorage,
};

/**
  | Similar with AliasOp, storing the alias
  | name as operator argument.
  |
  */
pub struct AliasWithNameOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    name:        String,
    is_backward: bool,
}

num_inputs!{AliasWithName, 1}

num_outputs!{AliasWithName, 1}

inputs!{AliasWithName, 
    0 => ("input", "Input tensor whose storage will be shared.")
}

outputs!{AliasWithName, 
    0 => ("output", "Tensor of same shape as input, sharing its storage.")
}

args!{AliasWithName, 
    0 => ("name", "name of the aliasing"),
    1 => ("is_backward", "weather or not to alias forward or backward")
}

identical_type_and_shape!{AliasWithName}

allow_inplace!{AliasWithName, vec![(0, 0)]}

impl<Context> AliasWithNameOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          name_(this->template GetSingleArgument<std::string>(
                  "name",
                  "invalid_name")),
                  is_backward_(
                      this->template GetSingleArgument<bool>("is_backward", false)) 

                      CAFFE_ENFORCE(
                          OperatorStorage::HasArgument("name"), "You have to specify argument name");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE_GE(input.numel(), 0, "Tensor is not initialized");

        // This doesn't work anymore as this is "newstyle" operator
        // OutputTensorAlias(0, input);

        OperatorStorage::SetOutputTensor(0, input.Alias());
        return true;
        */
    }
}

register_cpu_operator!{
    AliasWithName, 
    AliasWithNameOp<CPUContext>
}
