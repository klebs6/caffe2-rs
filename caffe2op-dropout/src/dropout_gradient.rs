crate::ix!();

///----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DropoutGradientOp<T,Context> {

    storage: OperatorStorage,
    context: Context,

    ratio:   f32,
    is_test: bool,

    /**
      | Input: dY, mask;
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<T>,
}

impl<T,Context> DropoutGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            ratio_(this->template GetSingleArgument<float>("ratio", 0.5)),
            is_test_(this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) 

        CAFFE_ENFORCE_GE(ratio_, 0);
        CAFFE_ENFORCE_LT(ratio_, 1);
        */
    }
}

register_cpu_gradient_operator!{
    DropoutGrad,
    DropoutGradientOp<f32, CPUContext>
}

num_inputs!{DropoutGrad, (1,2)}

num_outputs!{DropoutGrad, 1}

allow_inplace!{DropoutGrad, vec![(0, 0)]}

pub struct GetDropoutGradient;

impl GetGradientDefs for GetDropoutGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argshelper(def_);
        auto is_test = argshelper.GetSingleArgument<bool>("is_test", 0);
        if (is_test) {
          return SingleGradientDef(
              "DropoutGrad", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        } else {
          return SingleGradientDef(
              "DropoutGrad",
              "",
              vector<string>{GO(0), O(1)},
              vector<string>{GI(0)});
        }
        */
    }
}

register_gradient!{Dropout, GetDropoutGradient}
