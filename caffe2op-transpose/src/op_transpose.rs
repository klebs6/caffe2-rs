crate::ix!();

use crate::{
    OperatorDef,
    Tensor,
    GradientMakerBase,
    OperatorStorage,
};

#[test] fn transpose_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Transpose",
        ["X"],
        ["Y"],
        axes=(0,3,1,2)
    )

    x = np.random.rand(1,32,32,3)
    workspace.FeedBlob("X", x)
    print("X.shape (NHWC order):", workspace.FetchBlob("X").shape)
    workspace.RunOperatorOnce(op)
    print("Y.shape (NCHW order):", workspace.FetchBlob("Y").shape)

    X.shape (NHWC order): (1, 32, 32, 3)
    Y.shape (NCHW order): (1, 3, 32, 32)
    */
}

/**
  | Transpose the input tensor by permuting
  | the axes of the input according to the
  | `axes` argument.
  | 
  | Similar to numpy's [transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)
  | function.
  | 
  | For example, when axes=(1, 0, 2), given
  | an input tensor of shape (1, 2, 3), the
  | output shape will be (2, 1, 3).
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/transpose_op.cc
  |
  */
pub struct TransposeOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    //USE_DISPATCH_HELPER
    storage: OperatorStorage,
    context: Context,
    axes:    Vec<i32>,
}

register_cpu_operator!{Transpose, TransposeOp<CPUContext>}

num_inputs!{Transpose, 1}

num_outputs!{Transpose, 1}

inputs!{Transpose, 
    0 => ("X", "*(type: Tensor)* Input tensor.")
}

outputs!{Transpose, 
    0 => ("Y", "*(type: Tensor)* Transposed output.")
}

args!{Transpose, 
    0 => ("axes", "*(type: Tuple(int))* Order to permute axes of input tensor. Reverses the dimensions by default.")
}

tensor_inference_function!{
    Transpose, 
    /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      if (axes.empty()) {
        for (auto axis = in [0].dims().rbegin(); axis != in[0].dims().rend();
             ++axis) {
          out[0].add_dims(*axis);
        }
      } else {
        auto tensor_size = in[0].dims().size();
        auto valid_axes =
            std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
              return axis >= 0 && axis < tensor_size;
            });

        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
        CAFFE_ENFORCE(
            axes.size() == tensor_size,
            "Axes argument passed in had the incorrect size");

        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
          out[0].add_dims(in[0].dims().Get(*axis));
        }
      }

      return out;
    } */
}

inherit_onnx_schema!{Transpose}

impl<Context> TransposeOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes")) 

        // We will check the legality of axes_: it should be from 0 to axes_.size().
        std::vector<int> axes_sorted = axes_;
        std::sort(axes_sorted.begin(), axes_sorted.end());
        for (std::size_t i = 0; i < axes_sorted.size(); ++i) {
          if (axes_sorted[i] != i) {
            CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Do the actual transpose, which is implemented in DoRunWithType().
        return DispatchHelper<TensorTypes<float, double, int, int64_t>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn transpose_impl<T>(
        &mut self, 
        x: &Tensor,
        y: *mut Tensor)
    {
        todo!();
        /*
            const int ndim = X.dim();
        if (axes_.empty()) {
          axes_.resize(ndim);
          std::iota(axes_.rbegin(), axes_.rend(), 0);
        } else {
          CAFFE_ENFORCE_EQ(ndim, axes_.size());
        }
        const std::vector<std::int64_t> X_dims = X.sizes().vec();
        std::vector<std::int64_t> Y_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
          Y_dims[i] = X_dims[axes_[i]];
        }
        Y->Resize(Y_dims);
        math::Transpose<std::int64_t, T, Context>(
            X_dims.size(),
            X_dims.data(),
            axes_.data(),
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            TransposeImpl<T>(Input(0), Output(0));
        return true;
        */
    }
}

pub struct GetTransposeGradient;

impl CopyArguments for GetTransposeGradient {

    /// We will create our own arguments.
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

impl GetGradientDefs for GetTransposeGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            auto ops = SingleGradientDef(
            "Transpose", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        ops[0].mutable_arg()->CopyFrom(Def().arg());
        if (ArgumentHelper::HasArgument(Def(), "axes")) {
          // If axes is specified, we will need to figure out the inverse index.
          const Argument& old_axes = GetArgument(Def(), "axes");
          const int axes_size = old_axes.ints_size();
          Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
          for (int i = 0; i < axes_size; ++i) {
            new_arg->set_ints(old_axes.ints(i), i);
          }
        }
        return ops;
        */
    }
}

register_gradient!{Transpose, GetTransposeGradient}
