crate::ix!();

/**
 | Flattens the input tensor into a 2D matrix. If input tensor has shape
 | $(d_0, d_1, ..., d_n)$ then the output will have shape
 | $\bigl((d_0 * d_1 * ... * d_{(axis-1)}), (d_{axis} * d_{(axis+1)} * ... * d_n)\bigr)$.
 |
 | Github Links:
 |
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flatten_op.cc
 */
pub struct FlattenOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    axis:    i32,
}

num_inputs!{Flatten, 1}

num_outputs!{Flatten, 1}

inputs!{Flatten, 
    0 => ("X", "*(type: Tensor)* Input Tensor of rank >= axis.")
}

outputs!{Flatten, 
    0 => ("Y", "*(type: Tensor)* A 2D tensor with the contents of the input tensor, with input dimensions up to `axis` flattened to the outer dimension of the output and the remaining input dimensions flattened into the inner dimension of the output.")
}

args!{Flatten, 
    0 => ("axis", "*(type: int; default: 1)* Indicates up to which input dimensions (exclusive) should be flattened to the outer dimension of the output.")
}

tensor_inference_function!{Flatten, TensorInferenceForFlatten}

inherit_onnx_schema!{Flatten}

impl<Context> FlattenOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        CAFFE_ENFORCE_GE(
            input.dim(), axis_, "The rank of the tensor must be >= axis.");
        output->Resize(input.size_to_dim(axis_), input.size_from_dim(axis_));
        context_.CopyItemsSameDevice(
            input.dtype(),
            input.numel(),
            input.raw_data(),
            output->raw_mutable_data(input.dtype()));
        return true;
        */
    }
}

#[inline] pub fn tensor_inference_for_flatten(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int>("axis", 1);
      std::vector<TensorShape> out(1);
      int64_t outer = 1;
      int64_t inner = 1;
      std::size_t index = 0;
      for (auto d : in[0].dims()) {
        if (index < axis) {
          outer *= d;
        } else {
          inner *= d;
        }
        ++index;
      }
      out[0].set_data_type(in[0].data_type());
      out[0].add_dims(outer);
      out[0].add_dims(inner);
      return out;
    */
}

register_cpu_operator!{Flatten, FlattenOp<CPUContext>}

#[test] fn flatten_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Flatten",
        ["X"],
        ["Y"],
        axis=1
    )

    workspace.FeedBlob("X", np.random.rand(1,3,2,2))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[[[0.53432311 0.23734561]
       [0.56481598 0.52152617]]

      [[0.33662627 0.32472711]
       [0.17939016 0.97175851]]

      [[0.87226421 0.49045439]
       [0.92470531 0.30935077]]]]
    Y: [[0.53432311 0.23734561 0.56481598 0.52152617 0.33662627 0.32472711
      0.17939016 0.97175851 0.87226421 0.49045439 0.92470531 0.30935077]]
    */
}

pub struct GetFlattenGradient;

impl GetGradientDefs for GetFlattenGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{Flatten, GetFlattenGradient}
