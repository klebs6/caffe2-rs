crate::ix!();

/**
 | This operator takes in a 2d tensor, a list of
 | indices, and a 1d tensor with the same width of
 | the 2d tensor. It will replace the rows in 2d
 | tensor specified in indices with the 2d
 | tensor. The operator does an in-place change to
 | the input tensor.
 |
 |   Example:
 |     INPUT_TENSOR = [[1, 2], [3, 4], [5, 6]]
 |     INDICES = [1]
 |     ROW = [9, 0]
 |     OUTPUT_TENSOR = [[1, 2], [9, 0], [5, 6]]
 */
pub struct CopyRowsToTensorOp<Context> {
    storage: OperatorStorage,
    context: Context,

}

num_inputs!{CopyRowsToTensor, 3}

num_outputs!{CopyRowsToTensor, 1}

inputs!{CopyRowsToTensor, 
    0 => ("input_tensor", "Input tensor needs to be modified."),
    1 => ("indices",      "Indices of rows need to be copied"),
    2 => ("row",          "1-d tensor that is going to replace the rows")
}

outputs!{CopyRowsToTensor, 
    0 => ("output_tensor", "updated tensor")
}

tensor_inference_function!{CopyRowsToTensor,
    /*
    [](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0] = in[0];
      return out;
    }
    */
}

enforce_inplace!{CopyRowsToTensor, vec![(0, 0)]}

input_tags!{
    CopyRowsToTensorOp {
        InputTensor,
        Indices,
        Row
    }
}

impl<Context> CopyRowsToTensorOp<Context> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& input_tensor = Input(INPUT_TENSOR);
            auto& indices = Input(INDICES);
            auto& row = Input(ROW);
            auto tensor_width = input_tensor.size(1);
            CAFFE_ENFORCE_EQ(input_tensor.dim(), 2, "INPUT_TENSOR should be 2-d");
            CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES should be 1-d");
            CAFFE_ENFORCE_EQ(row.dim(), 1, "ROW should be 1-d");
            CAFFE_ENFORCE_EQ(
                tensor_width,
                row.size(0),
                "width of input tensor should match lengths of row");
            const auto* indices_data = indices.template data<int64_t>();
            const auto* row_data = row.template data<T>();
            auto* output = Output(0);
            auto* output_data = output->template mutable_data<T>();
            CAFFE_ENFORCE(
                IsInputOutputAlias(0, 0), "Input 0 and Output 0 should be alias.");
            for (size_t i = 0; i < indices.sizes()[0]; ++i) {
              std::memcpy(
                  output_data + indices_data[i] * tensor_width,
                  row_data,
                  tensor_width * sizeof(T));
            }
            return true;
        */
    }
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<at::Half, float, double, int32_t, int64_t>>::
            call(this, Input(INPUT_TENSOR));
        */
    }
}
