crate::ix!();

/**
  | DenseVectorToIdList: Convert a blob
  | with dense feature into a ID_LIST.
  | 
  | An ID_LIST is a list of IDs (may be ints,
  | often longs) that represents a single
  | feature. As described in https://caffe2.ai/docs/sparse-operations.html,
  | a batch of ID_LIST examples is represented
  | as a pair of lengths and values where
  | the `lengths` (int32) segment the `values`
  | or ids (int32/int64) into examples.
  | 
  | Input is a single blob where the first
  | dimension is the batch size and the second
  | dimension is the length of dense vectors.
  | This operator produces a
  | 
  | ID_LIST where out_values are the indices
  | of non-zero entries and out_lengths
  | are the number of non-zeros entries
  | in each row.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DenseVectorToIdListOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{DenseVectorToIdList, 1}

num_outputs!{DenseVectorToIdList, 2}

inputs!{DenseVectorToIdList, 
    0 => ("values", "A data blob of dense vectors")
}

outputs!{DenseVectorToIdList, 
    0 => ("out_lengths", "Lengths of the sparse feature"),
    1 => ("out_values", "Values of the sparse feature")
}

register_cpu_operator!{
    DenseVectorToIdList, 
    DenseVectorToIdListOp<CPUContext>
}

no_gradient!{DenseVectorToIdList}

impl<Context> DenseVectorToIdListOp<Context> {

    #[inline] pub fn do_run_with_type<T, M>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
            const auto* input_data = input.template data<T>();

            CAFFE_ENFORCE_EQ(input.dim(), 2, "Sample should be 2-D");
            const auto batch_size = input.size(0);
            const auto col_num = input.size(1);

            auto* out_lengths = Output(0, {batch_size}, at::dtype<int32_t>());

            auto* out_lengths_data = out_lengths->template mutable_data<int32_t>();

            auto* out_values = Output(1, {batch_size * col_num}, at::dtype<M>());

            auto* out_values_data = out_values->template mutable_data<M>();

            auto v_pos = 0;
            auto l_pos = 0;
            for (auto i = 0; i < batch_size; i++) {
              auto length = 0;
              for (int j = 0; j < col_num; j++) {
                if ((int)(input_data[i * col_num + j] + 0.5) != 0) {
                  out_values_data[v_pos++] = j;
                  length++;
                }
              }
              out_lengths_data[l_pos++] = length;
            }
            out_values->Resize(v_pos);
            out_lengths->Resize(l_pos);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).template IsType<float>()) {
          return DoRunWithType<float, int>();
        } else {
          CAFFE_THROW(
              "DenseVectorToIdList operator only supports 32-bit float, but",
              " input was of type ",
              Input(0).dtype().name());
        }
        */
    }
}
