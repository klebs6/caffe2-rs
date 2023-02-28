crate::ix!();

use crate::{
    CPUContext,
    OperatorDef,
    GradientMakerBase,
};

/**
  | Softsign takes one input data tensor
  | $X$ and produces one output data $Y,$
  | where the softsign function, $y = \frac{x}{1+
  | |x|}$, is applied to $X$ elementwise.
  | 
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc
  |
  */
pub struct SoftsignFunctor<Context> { 

    phantom: PhantomData<Context>,
}

num_inputs!{Softsign, 1}

num_outputs!{Softsign, 1}

inputs!{Softsign, 
    0 => ("input", "Input data blob to be operated on.")
}

outputs!{Softsign, 
    0 => ("output", "Output data blob with same shape as input")
}

identical_type_and_shape!{Softsign}

allow_inplace!{Softsign, vec![(0, 0)]}

inherit_onnx_schema!{Softsign}

#[test] fn softsign_functor_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Softsign",
        ["X"],
        ["Y"],
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[-1.3060539   0.7242748  -1.9907674 ]
     [-0.64802396 -0.03244735  0.7455406 ]
     [-0.298492   -0.5774271   2.8364444 ]]

    Y:
     [[-0.5663588   0.420046   -0.6656376 ]
     [-0.39321268 -0.03142761  0.4271116 ]
     [-0.2298759  -0.36605626  0.739342  ]]

    */
}

impl SoftsignFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            ConstEigenVectorArrayMap<T> X_arr(X, N);
      EigenVectorMap<T>(Y, N) = (T(1) + X_arr.abs()).inverse() * X_arr;
      return true;
        */
    }
}

/**
  | Calculates the softsign gradient (sgn(x)/(1+|x|)^2)
  | of the given input tensor element-wise.
  |
  */
pub struct SoftsignGradientFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{SoftsignGradient, 2}

num_outputs!{SoftsignGradient, 1}

inputs!{SoftsignGradient, 
    0 => ("input", "1-D input tensor"),
    1 => ("input", "1-D input tensor")
}

outputs!{SoftsignGradient, 
    0 => ("output", "The softsign gradient (sgn(x)/(1+|x|)^2) 
        values of the input tensor computed element-wise")
}

allow_inplace!{SoftsignGradient, vec![(1, 0)]}

impl SoftsignGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        x:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int size = std::accumulate(
          X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
      ConstEigenVectorArrayMap<T> dY_arr(dY, size);
      ConstEigenVectorArrayMap<T> X_arr(X, size);
      EigenVectorMap<T>(dX, size) =
          dY_arr * (T(1) + X_arr.abs()).square().inverse();
      return true;
        */
    }
}

register_cpu_operator!{
    Softsign,
    UnaryElementwiseOp<
        TensorTypes::<f32>,
        CPUContext,
        SoftsignFunctor::<CPUContext>>
}

register_cpu_gradient_operator!{
    SoftsignGradient,
    BinaryElementwiseOp::<
        TensorTypes::<f32>,
        CPUContext,
        SoftsignGradientFunctor::<CPUContext>>}

pub struct GetSoftsignGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSoftsignGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            I(0) != O(0),
            "Cannot compute softsign gradient "
            "if you choose to do an in-place calculation.");

        return SingleGradientDef(
            "SoftsignGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Softsign, GetSoftsignGradient}
