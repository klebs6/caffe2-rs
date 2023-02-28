crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
};

#[test] fn expand_dims_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

        op = core.CreateOperator(
            "ExpandDims",
            ["data"],
            ["expanded"],
            dims=[0,1],
        )

        workspace.FeedBlob("data", np.zeros((100,100)).astype(np.float32))
        print("data.shape:", workspace.FetchBlob("data").shape)

        workspace.RunOperatorOnce(op)
        print("expanded.shape:", workspace.FetchBlob("expanded").shape)


        data.shape: (100, 100)
        expanded.shape: (1, 1, 100, 100)
    */
}

/**
  | The *ExpandDims* op inserts single-dimensional
  | entries into the shape of the input tensor
  | *data,* and produces a single output
  | tensor *expanded*.
  | 
  | The op also takes an argument *dims*
  | with a list of dimensions for where to
  | add the single dimensional entries.
  | 
  | If the same blob is provided as input
  | and output, the operation is copy-free.
  | This is the exact inverse operation
  | of *Squeeze*.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc
  |
  */
pub struct ExpandDimsOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    dims:    Vec<i32>,
}

num_inputs!{ExpandDims, 1}

num_outputs!{ExpandDims, 1}

inputs!{ExpandDims, 
    0 => ("data", "Input tensor of data to be operated on.")
}

outputs!{ExpandDims, 
    0 => ("expanded", "Reshaped tensor with same data as input.")
}

args!{ExpandDims, 
    0 => ("dims", "*(type: [int])* List of dimensions of *data* to add single dimensional entry.")
}

inherit_onnx_schema!{ExpandDims}

tensor_inference_function!{ExpandDims, 
    /*[](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }

      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");
      CAFFE_ENFORCE_GE(
          in[0].dims_size() + dims.size(),
          dims.back() + 1,
          "Input needs at least ",
          (1 + dims.back() - dims.size()),
          " dimensions given `dims`.");

      vector<TensorShape> out(1);

      int cur_pos = 0;
      int idx = 0;
      for (const auto new_dim : dims) {
        for (int i = cur_pos; i < new_dim; i++) {
          out[0].add_dims(in[0].dims(idx++));
        }
        out[0].add_dims(1);
        cur_pos = new_dim + 1;
      }
      for (; idx < in[0].dims_size(); idx++) {
        out[0].add_dims(in[0].dims(idx));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    }*/
}

allow_inplace!{ExpandDims, vec![(0, 0)]}

impl<Context> ExpandDimsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dims_(this->template GetRepeatedArgument<int>("dims")) 

        auto originalSize = dims_.size();
        CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");
        std::sort(dims_.begin(), dims_.end());
        dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
        if (dims_.size() < originalSize) {
          LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
        }
        CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        output->CopyFrom(input, true /*async*/);
        if (dims_.empty()) {
          return true;
        }

        auto newDims = input.sizes().vec();
        CAFFE_ENFORCE_GE(
            input.sizes().size() + dims_.size(),
            dims_.back() + 1,
            "Input needs at least ",
            (1 + dims_.back() - dims_.size()),
            " dimensions given `dims`.");
        for (const auto dim : dims_) {
          newDims.insert(newDims.begin() + dim, 1);
        }
        output->Reshape(newDims);
        return true;
        */
    }
}

///-------------------------------------------

#[test] fn squeeze_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Squeeze",
        ["data"],
        ["squeezed"],
        dims=[0,1],
    )

    workspace.FeedBlob("data", np.zeros((1,1,100,100)).astype(np.float32))
    print("data.shape:", workspace.FetchBlob("data").shape)

    workspace.RunOperatorOnce(op)
    print("squeezed.shape:", workspace.FetchBlob("squeezed").shape)

    data.shape: (1, 1, 100, 100)
    squeezed.shape: (100, 100)
    */
}

/**
  | The *Squeeze* op removes single-dimensional
  | entries from the shape of the input tensor
  | *data,* and produces a single output
  | tensor *squeezed*.
  | 
  | The op also takes an argument *dims*
  | with a list of dimensions to squeeze.
  | 
  | If the same blob is provided as input
  | and output, the operation is copy-free.
  | 
  | This is the exact inverse operation
  | of
  | 
  | ExpandDims* given the same *dims* argument.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc
  |
  */
pub struct SqueezeOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    dims:    Vec<i32>,
}

num_inputs!{Squeeze, 1}

num_outputs!{Squeeze, 1}

inputs!{Squeeze, 
    0 => ("data", "Input tensor of data to be operated on.")
}

outputs!{Squeeze, 
    0 => ("squeezed", "Reshaped tensor with same data as input.")
}

args!{Squeeze, 
    0 => ("dims", "*(type: [int])* List of dimensions of *data* to squeeze out.")
}

allow_inplace!{Squeeze, vec![(0, 0)]}

inherit_onnx_schema!{Squeeze}

tensor_inference_function!{Squeeze, /*[](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }
      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");

      vector<TensorShape> out(1);
      std::vector<int> newDims =
          SqueezeOp<CPUContext>::ComputeDims(GetDimsVector(in[0]), dims);
      out[0] = CreateTensorShape(newDims, in[0].data_type());
      return out;
    }*/
}

impl<Context> SqueezeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dims_(this->template GetRepeatedArgument<int>("dims")) 

        auto originalSize = dims_.size();
        CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

        std::sort(dims_.begin(), dims_.end());
        dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
        if (dims_.size() < originalSize) {
          LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
        }
        CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        output->CopyFrom(input, true /*async*/);

        CAFFE_ENFORCE_GT(
            input.dim(),
            dims_.back(),
            "Input needs at least ",
            (dims_.back() + 1),
            " dimensions.");

        std::vector<int> newDims = ComputeDims(input.sizes(), dims_);
        output->Reshape(newDims);
        return true;
        */
    }
    
    #[inline] pub fn compute_dims(input_dims: &[i32], dims: Vec<i32>) -> Vec<i32> {
        
        todo!();
        /*
            size_t j = 0;
        std::vector<int> newDims;
        for (size_t i = 0; i < inputDims.size(); ++i) {
          if (j < dims.size() && dims[j] == i) {
            CAFFE_ENFORCE_EQ(
                inputDims[i],
                1,
                "Dimension ",
                i,
                " of input must be 1",
                " instead of ",
                inputDims[i],
                ".");
            ++j;
            continue;
          }
          newDims.push_back(inputDims.at(i));
        }
        return newDims;
        */
    }
}

register_cpu_operator!{ExpandDims, ExpandDimsOp<CPUContext>}

register_cpu_operator!{Squeeze,    SqueezeOp<CPUContext>}

///--------------------
pub struct GetSqueezeGradient;

impl GetGradientDefs for GetSqueezeGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ExpandDims", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{Squeeze, GetSqueezeGradient}

register_cuda_operator!{Squeeze, SqueezeOp<CUDAContext>}

///--------------------
pub struct GetExpandDimsGradient;

impl GetGradientDefs for GetExpandDimsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Squeeze", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ExpandDims, GetExpandDimsGradient}

register_cuda_operator!{ExpandDims, ExpandDimsOp<CUDAContext>}
