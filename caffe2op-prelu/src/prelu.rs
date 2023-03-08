crate::ix!();

/**
  | The *PRelu* op takes input data tensor
  | $X$, an input slope tensor $slope$,
  | and produces one output tensor $Y$ of
  | the same shape as $X.$ The op performs
  | the element wise *PRelu* operation,
  | defined as
  | 
  | $$y=prelu(x) =\begin{cases}slope
  | * x & x < 0\\x & otherwise\end{cases}$$
  | 
  | Note, is slope is size 1, the value is
  | shared across the channels, otherwise
  | $X$ and $slope$ must be the same shape.
  | See [Delving Deep into Rectifiers:
  | Surpassing Human-Level Performance
  | on
  | 
  | ImageNet Classification](https://arxiv.org/abs/1502.01852)
  | for more information.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.h
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PReluOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    order:   StorageOrder,

    /// Input: X, Slope, output: Y
    phantom: PhantomData<T>,
}

impl<T,Context> PReluOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<string>("order", "NCHW")))
        */
    }
}
