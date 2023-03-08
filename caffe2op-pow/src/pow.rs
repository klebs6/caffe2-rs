crate::ix!();

pub type FactorThisShitOut = i32;
pub type SameTypeAsInput   = FactorThisShitOut;

/**
  | The *Pow* op takes an input data tensor
  | $X$ and an exponent parameter *exponent*,
  | which can be a scalar or another tensor.
  | As output, it produces a single output
  | data tensor $Y$, where the function
  | $f(x) = x^{exponent}$ has been applied
  | to $X$ elementwise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PowOp<InputTypes,Context,Functor,TypeMap = SameTypeAsInput> {
    storage:           OperatorStorage,
    context:           Context,
    enable_broadcast:  bool,
    axis:              i32,
    axis_str:          String,
    order:             String,
    exponent:          f32,
    functor:           Functor,
    phantom:           PhantomData<InputTypes>,
    phantomTypeMap:    PhantomData<TypeMap>,
}
