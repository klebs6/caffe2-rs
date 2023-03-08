crate::ix!();

use crate::{
    OperatorStorage
};

#[test] fn shape_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Shape",
        ["X"],
        ["shape"],
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(2,3))))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("shape:", workspace.FetchBlob("shape"))

    X:
    [[3 2 5]
     [5 7 3]]
    shape: [2 3]
    */
}

/**
  | Produce a 1D int64 tensor with the shape
  | of the input tensor.
  | 
  | If called with an optional argument
  | `axes`, the result will only contain
  | the dimensions of specified axes.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/shape_op.cc
  | 
  | RecordShapeOp records the shape of
  | the input tensor to a vector of int. You
  | mostly don't need this operator explicitly,
  | and it is mostly used in the autodiff
  | process.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ShapeOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axes:    Vec<i32>,
}

register_cpu_operator!{Shape, ShapeOp<CPUContext>}

num_inputs!{Shape, 1}

num_outputs!{Shape, 1}

inputs!{Shape, 
    0 => ("X", "*(type: Tensor)* Input tensor.")
}

outputs!{Shape, 
    0 => ("shape", "*(type: Tensor)* Output tensor containing shape of input tensor.")
}

args!{Shape, 
    0 => ("axes", "*(type: int[])* Array of interested axes. If given, this operator only returns the dimensions of the given axes. Otherwise, the operator returns the dimensions of all axes.")
}

tensor_inference_function!{Shape, /* ([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper args(def);
      const vector<int>& axes = args.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      if (axes.empty()) {
        out[0].add_dims(in[0].dims().size());
      } else {
        out[0].add_dims(axes.size());
      }
      out[0].set_data_type(TensorProto::INT64);
      return out;
    }) */}

should_not_do_gradient!{Shape}

register_cuda_operator!{Shape, ShapeOp<CUDAContext>}

input_tags!{
    ShapeOp {
        Data
    }
}

impl<Context> ShapeOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(OperatorStorage ::GetRepeatedArgument<int>("axes"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(DATA);

        int numDims = data.dim();
        int numAxes = axes_.size();
        if (numAxes == 0) {
          auto* output = Output(0, {numDims}, at::dtype<int64_t>());
          int64_t* output_data = output->template mutable_data<int64_t>();
          context_.CopyBytesSameDevice(
              numDims * sizeof(int64_t), data.sizes().data(), output_data);
          return true;
        }

        auto* output = Output(0, {numAxes}, at::dtype<int64_t>());
        auto src = reinterpret_cast<const char*>(data.sizes().data());
        auto out = reinterpret_cast<char*>(output->template mutable_data<int64_t>());
        for (int i = 0; i < numAxes; i++) {
          auto axis = axes_[i];
          CAFFE_ENFORCE_LT(axis, numDims, "Axis out of range");
          CAFFE_ENFORCE_GE(axis, 0, "Each axis should be non-negative");
          context_.CopyBytesSameDevice(
              sizeof(int64_t), src + axis * sizeof(int64_t), out);
          out += sizeof(int64_t);
        }
        return true;
        */
    }
}
