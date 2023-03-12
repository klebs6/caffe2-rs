crate::ix!();

/**
 | SumReduceLike operator takes 2 tensors as
 | input. It performs reduce sum to the first input
 | so that the output looks like the second one.
 |
 | It assumes that the first input has more
 | dimensions than the second, and the dimensions of
 | the second input is the contiguous subset of the
 | dimensions of the first.
 |
 | For example, the following tensor shapes are
 | supported:
 |
 |   shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
 |   shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
 |   shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
 |   shape(A) = (2, 3, 2, 5), shape(B) = (2), with axis=0
 |
 | Sum reduction operator that is used for computing
 | the gradient in cases where the forward op is in
 | broadcast mode.
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumReduceLikeOp<Context> {

    storage: OperatorStorage,
    context: Context,

    axis:     i32,
    axis_str: String,
    order:    String,

    ones:       Tensor, // {Context::GetDeviceType()};
    sum_buffer: Tensor, // {Context::GetDeviceType()};
}

num_inputs!{SumReduceLike, 2}

num_outputs!{SumReduceLike, 1}

inputs!{SumReduceLike, 
    0 => ("A", "First operand, should share the type with the second operand."),
    1 => ("B", "Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.")
}

outputs!{SumReduceLike, 
    0 => ("C", "Result, has same dimensions and type as B")
}

args!{SumReduceLike, 
    0 => ("axis",      "If set, defines the starting dimension for reduction. Args `axis` and `axis_str` cannot be used simultaneously."),
    1 => ("axis_str",  "If set, it could only be N or C or H or W. `order` arg should also be provided. It defines the reduction dimensions on NCHW or NHWC. Args `axis` and `axis_str` cannot be used simultaneously."),
    2 => ("order",     "Either NHWC or HCWH")
}

identical_type_and_shape_of_input!{SumReduceLike, 0}

impl<Context> SumReduceLikeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
            OP_SINGLE_ARG(string, "order", order_, "NCHW") 

        if (axis_ != -1) {
          // Get axis from an explicit axis argument.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(),
              0U,
              "Args axis and axis_str cannot be used simultaneously.");
        } else if (axis_str_.size()) {
          // Get the axis index semantically.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
          size_t semantic_axis = order_.find(axis_str_);
          CAFFE_ENFORCE_NE(
              semantic_axis,
              string::npos,
              "Unrecognizable axis string ",
              axis_str_,
              " from order string ",
              order_);
          axis_ = semantic_axis;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
}

register_cpu_operator!{
    Not,
    UnaryElementwiseOp<BoolTypes, CPUContext, NotFunctor<CPUContext>>
}

register_cpu_operator!{
    Sign,
    UnaryElementwiseOp<NumericTypes, CPUContext, SignFunctor<CPUContext>>
}

pub mod srl_helper {

    use super::*;

    #[inline] pub fn sum2one<T>(x: *const T, y: *mut T, n: usize) {
        todo!();
        /*
            *y = ConstEigenArrayMap<T>(x, n, 1).sum();
        */
    }

    #[inline] pub fn run_with_broadcast_front<T>(
        x: *const T,
        y: *mut T,
        pre: usize,
        n: usize,
        context: *mut CPUContext) 
    {
        todo!();
        /*
            EigenArrayMap<T>(y, n, 1) = ConstEigenArrayMap<T>(x, n, pre).rowwise().sum();
        */
    }


    #[inline] pub fn run_with_broadcast_back<T>(
        x:       *const T,
        y:       *mut T,
        post:    usize,
        n:       usize,
        context: *mut CPUContext) 
    {
        todo!();
        /*
            EigenArrayMap<T>(y, 1, n) = ConstEigenArrayMap<T>(x, post, n).colwise().sum();
        */
    }

    #[inline] pub fn run_with_broadcast2<T>(
        a:        *const T,
        y:        *mut T,
        pre:      usize,
        n:        usize,
        post:     usize,
        context:  *mut CPUContext) 
    {
        todo!();
        /*
            for (auto i = 0U; i < n; ++i) {
            y[i] = 0;
            for (auto j = 0U; j < pre; ++j) {
              for (auto k = 0U; k < post; ++k) {
                y[i] += a[(j * n + i) * post + k];
              }
            }
          }
        */
    }
}
