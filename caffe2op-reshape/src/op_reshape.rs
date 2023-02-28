crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
    OperatorDef,
    GradientMakerBase
};

#[test] fn reshape_op_example() {

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Reshape",
        ["data"],
        ["reshaped", "old_shape"],
        shape=(3,2)
    )

    workspace.FeedBlob("data", (np.random.randint(100, size=(6))))
    print("data:", workspace.FetchBlob("data"))
    workspace.RunOperatorOnce(op)
    print("reshaped:", workspace.FetchBlob("reshaped"))
    print("old_shape:", workspace.FetchBlob("old_shape"))

    data: [86 60 85 96  7 37]
    reshaped: [[86 60]
              [85 96]
              [ 7 37]]
    old_shape: [6]
    */
}

/**
  | Takes a shape and data tensor and reshapes
  | it
  | 
  | Reshape the input tensor similar to
  | numpy's [reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html).
  | 
  | Takes a tensor as input and an optional
  | tensor specifying the new shape. When
  | the second input is absent, an extra
  | argument shape must be specified. Outputs
  | the reshaped tensor as well as the original
  | shape.
  | 
  | At most one dimension of the new shape
  | can be -1. In this case, the value is inferred
  | from the size of the tensor and the remaining
  | dimensions. A dimension could also
  | be 0, in which case the actual dimension
  | value is going to be copied from the input
  | tensor.
  | 
  | For empty tensor, we will set the -1 dimension
  | to be 0 (if one dimension is -1).
  | 
  | When the tensor is empty, dimension
  | of 0 will remain to be 0.
  | 
  | E.g: data=np.empty(shape=[4, 0]),
  | shape=[0, -1], the output tensor will
  | be np.emtpy(shape=[0, 0])
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reshape_op.cc
  |
  */
pub struct ReshapeOp<F,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:   OperatorStorage,
    context:   Context,
    new_shape: Vec<i64>,
    phantomF:  PhantomData<F>,
}

register_cpu_operator!{Reshape, ReshapeOp<f32, CPUContext>}

register_cuda_operator!{Reshape, ReshapeOp<float, CUDAContext>}

num_inputs!{Reshape, (1,2)}

num_outputs!{Reshape, 2}

inputs!{Reshape, 
    0 => ("data", "*(type: Tensor)* Input tensor."),
    1 => ("new_shape", "*(type: Tensor`<int>`)* [OPTIONAL] Tensor containing new shape.")
}

outputs!{Reshape, 
    0 => ("reshaped", "*(type: Tensor)* Reshaped output tensor."),
    1 => ("old_shape", "*(type: Tensor`<int>`)* Tensor containing old shape of `data`.")
}

args!{Reshape, 
    0 => ("shape", "*(type: Tuple(int))* New shape. Do not set if using `new_shape` input.")
}

tensor_inference_function!{Reshape, /* ([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(2);

      // Do shape inference for old_shape
      out[1].set_data_type(TensorProto::INT64);
      out[1].add_dims(in[0].dims_size());

      ArgumentHelper helper(def);
      if (!helper.HasArgument("shape")) {
        // Cannot do shape inference for reshaped tensor from runtime data.
        CAFFE_ENFORCE_EQ(
            in.size(),
            2,
            "New shape must be specified by either the input blob or the "
            "argument `shape`.");
        out[0].set_unknown_shape(true);
        return out;
      }
      CAFFE_ENFORCE_EQ(
          in.size(),
          1,
          "New shape must not be specified by the input blob and the "
          "argument `shape` at the same time.");

      // Infer the actual new shape
      auto actualNewShape = helper.GetRepeatedArgument<int64_t>("shape");

      // Copy over the dimensions for those that are specified zero
      // and check the eligibility of input
      for (int i = 0; i < actualNewShape.size(); ++i) {
        CAFFE_ENFORCE_GE(
            actualNewShape[i],
            -1,
            "The dimensions in argument `shape` "
            "must not be a negative number.");

        if (actualNewShape[i] == 0) {
          CAFFE_ENFORCE_LT(
              i,
              in[0].dims_size(),
              "Argument `shape` has a dimension set to zero that exceeds "
              "the original dimension size.");
          actualNewShape[i] = in[0].dims(i);
        }
      }

      // Check if the new shape is valid and fills in the missing dimension
      // specified by -1.
      int64_t totalSize = 1;
      for (const auto d : in[0].dims()) {
        totalSize *= d;
      }
      int64_t size = 1;
      int unknownIdx = -1;
      for (int i = 0; i < actualNewShape.size(); ++i) {
        const auto dim = actualNewShape[i];
        if (dim == -1) {
          CAFFE_ENFORCE(
              unknownIdx == -1,
              "Argument `shape` has more than one missing dimension.");
          unknownIdx = i;
        } else {
          size *= dim;
        }
      }

      if (unknownIdx != -1) {
        CAFFE_ENFORCE(
            totalSize % size == 0,
            "Argument `shape` does not agree with the input data.",
            " (",
            totalSize,
            " vs ",
            size,
            ")");
        actualNewShape[unknownIdx] = totalSize / size;
      } else {
        CAFFE_ENFORCE_EQ(
            totalSize,
            size,
            "Argument `shape` does not agree with the input data.",
            " (",
            totalSize,
            " != ",
            size,
            ")");
      }

      out[0].set_data_type(in[0].data_type());
      for (const auto d : actualNewShape) {
        out[0].add_dims(d);
      }
      return out;
    }) */}

allow_inplace!{Reshape, vec![(0, 0)]}

inherit_onnx_schema!{Reshape}

impl<F,Context> ReshapeOp<F,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            new_shape_(this->template GetRepeatedArgument<int64_t>("shape"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() == 2) {
          return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
        }
        CAFFE_ENFORCE(
            OperatorStorage::HasArgument("shape"), "Argument `shape` is missing.");
        return this->template DoRunWithType<int64_t>();
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            DoRunWithTypeImpl<T>(Input(0), Output(0));
        return true;
        */
    }
    
    #[inline] pub fn do_run_with_type_impl<T>(&mut self, input: &Tensor, output: *mut Tensor)  {
    
        todo!();
        /*
            vector<int64_t> actual_new_shape = new_shape_;
        if (InputSize() == 2) {
          CAFFE_ENFORCE(
              !OperatorStorage::HasArgument("shape"),
              "New shape is specified by the input blob, do not pass in "
              "the argument `shape`.");

          // Shape should be always stored only on CPU
          // Just in case if for some reason shape is on GPU
          if (this->InputIsTensorType(1, CPU)) {
            // originally, shape input must be in CPU context
            auto& shape = this->template Input<Tensor>(1, CPU);
            CAFFE_ENFORCE_EQ(
                shape.dim(),
                1,
                "When input_as_shape is true, the input must be a 1D tensor of "
                "data type int64_t");
            CAFFE_ENFORCE(shape.numel() > 0);
            auto* shape_data = shape.template data<T>();
            actual_new_shape.insert(
                actual_new_shape.end(), shape_data, shape_data + shape.dim32(0));
          } else {
            auto& shape = Input(1);
            CAFFE_ENFORCE_EQ(
                shape.dim(),
                1,
                "When input_as_shape is true, the input must be a 1D tensor of "
                "data type int64_t");
            CAFFE_ENFORCE(shape.numel() > 0);
            auto* shape_data = shape.template data<T>();
            // Fetch copy from
            std::unique_ptr<T[]> shape_data_copy =
                std::make_unique<T[]>(shape.dim32(0));
            context_.template CopyToCPU<T>(
                shape.dim32(0), shape_data, shape_data_copy.get());
            actual_new_shape.insert(
                actual_new_shape.end(),
                shape_data_copy.get(),
                shape_data_copy.get() + shape.dim32(0));
          }
        }

        // Checks if the new shape is valid and fills in the missing dimension
        // specified by -1.
        // NOTE: At most one dimension can be -1.
        auto total_size = input.numel();
        T size = 1;

        // NOTE: support for legacy caffe1 syntax
        // Copy over the dimensions for those that are specified zero.
        if (total_size != 0) {
          for (size_t i = 0; i < actual_new_shape.size() && i < input.dim(); ++i) {
            if (actual_new_shape[i] == 0) {
              actual_new_shape[i] = input.size(i);
            }
          }
        }

        int unknown_idx = -1;
        for (int i = 0; i < actual_new_shape.size(); ++i) {
          const auto dim = actual_new_shape[i];
          if (dim == -1) {
            CAFFE_ENFORCE(
                unknown_idx == -1,
                "Argument `shape` has more than one missing dimension.");
            unknown_idx = i;
          } else {
            size *= dim;
          }
        }
        if (size == 0 && total_size != 0) {
          CAFFE_THROW(
              "Can not reshape a non-zero size (",
              total_size,
              ") tensor to zero size.");
        }
        if (total_size != 0) {
          // if tensor is not empty, infer the size of the unknown index
          if (unknown_idx != -1) {
            CAFFE_ENFORCE_NE(
                size,
                0,
                "New shape at dim ",
                unknown_idx,
                " can not be inferred since new size is zero.");
            CAFFE_ENFORCE(
                total_size % size == 0,
                "Argument `shape` does not agree with the input data.",
                " (",
                total_size,
                " vs ",
                size,
                ")");
            actual_new_shape[unknown_idx] = total_size / size;
          } else {
            CAFFE_ENFORCE_EQ(
                total_size,
                size,
                "Argument `shape` does not agree with the input data.",
                " (",
                total_size,
                " != ",
                size,
                ")");
          }
        } else if (unknown_idx != -1) {
          // if size is empty, then set unknown index to be 0 (empty tensor)
          actual_new_shape[unknown_idx] = 0;
        }

        // Write the original shape to the second output.
        auto* old_shape = this->template Output<Tensor>(1, CPU);
        old_shape->Resize(input.sizes().size());
        T* old_shape_data = old_shape->template mutable_data<T>();
        std::vector<T> old_shape_vector(input.sizes().begin(), input.sizes().end());
        for (int i = 0; i < old_shape_vector.size(); ++i) {
          old_shape_data[i] = old_shape_vector[i];
        }

        output->Resize(actual_new_shape);
        if (output != &input) {
          // If we are not doing in-place computation, a copy is needed.
          context_.CopyItemsSameDevice(
              input.dtype(),
              input.numel(),
              input.raw_data(),
              output->raw_mutable_data(input.dtype()));
        }
        */
    }
}

///--------------------
pub struct GetReshapeGradient {

}

impl GetGradientDefs for GetReshapeGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Reshape",
            "",
            vector<string>{GO(0), O(1)},
            vector<string>{GI(0), "_" + GI(0) + "_dims"});
        */
    }
}

impl CopyArguments for GetReshapeGradient {

    /**
      | Argument `shape` is no longer needed
      | in backprop.
      |
      */
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{Reshape, GetReshapeGradient}

