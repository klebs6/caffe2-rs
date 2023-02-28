crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
    CPUContext,
};

/**
  | Greedy decoder for connectionist temporal
  | classification.
  |
  */
pub struct CTCGreedyDecoderOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    merge_repeated: bool,

    /*
      | Input: X, 3D tensor; L, 1D tensor.
      | 
      | Output: Y sparse tensor
      |
      */
}

register_cpu_operator!{
    CTCGreedyDecoder, 
    CTCGreedyDecoderOp<CPUContext>
}

num_inputs!{CTCGreedyDecoder, (1,2)}

num_outputs!{CTCGreedyDecoder, 2}

inputs!{CTCGreedyDecoder, 
    0 => ("INPUTS", "3D float Tensor sized [max_time, batch_size, num_classes]"),
    1 => ("SEQ_LEN", "(optional) 1D int vector containing sequence lengths, having size [batch_size] seq_len will be set to max_time if not provided")
}

outputs!{CTCGreedyDecoder, 
    0 => ("OUTPUT_LEN", "Output_len matrix size (batch). The row store: [decoded_length]"),
    1 => ("VALUES", "Values vector, size (total_decoded_outputs). The vector stores the decoded classes")
}

args!{CTCGreedyDecoder, 
    0 => ("merge_repeated", "When merge_repeated is true, merge repeated classes in output.")
}

inherit_onnx_schema!{CTCGreedyDecoder}

should_not_do_gradient!{CTCGreedyDecoder}

input_tags!{
    CTCGreedyDecoderOp {
        Inputs,
        SeqLen
    }
}

output_tags!{
    CTCGreedyDecoderOp {
        OutputLen,
        Values
    }
}

impl<Context> CTCGreedyDecoderOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 
        merge_repeated_ =
            this->template GetSingleArgument<bool>("merge_repeated", true);
        */
    }
}

#[inline] pub fn get_tensor_data_ptr(tensor: &Tensor, t: i32, n: i32) -> *const f32 {
    
    todo!();
    /*
        const auto dims = tensor.sizes();
      CAFFE_ENFORCE_EQ(dims.size(), 3);
      int64_t offset = (t * dims[1] + n) * dims[2];
      CAFFE_ENFORCE_LT(offset, tensor.numel());
      return tensor.template data<float>() + offset;
    */
}

impl CTCGreedyDecoderOp<CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // [max_time_step, batch_size, num_classes]
      auto& inputs = Input(INPUTS);
      // [batch_size]

      // [total_decoded_output]

      const auto inputs_dims = inputs.sizes();
      int32_t max_time_step = inputs_dims[0];
      int32_t batch_size = inputs_dims[1];
      int32_t num_classes = inputs_dims[2];
      // [batch_size]
      const int* seq_len_data =
          (InputSize() == 2) ? Input(SEQ_LEN).data<int>() : nullptr;

      vector<int> values_cach;
      auto* output_len =
          Output(OUTPUT_LEN, vector<int64_t>{batch_size}, at::dtype<int>());
      int* output_len_data = output_len->template mutable_data<int>();

      for (int32_t i = 0; i < batch_size; ++i) {
        int previous_label = 0, t_dec = 0;
        int32_t seq_len_i = (seq_len_data) ? seq_len_data[i] : max_time_step;
        CAFFE_ENFORCE_LE(seq_len_i, max_time_step);
        for (int32_t t = 0; t < seq_len_i; ++t) {
          auto* prob_data = getTensorDataPtr(inputs, t, i);
          int curr_label =
              std::max_element(prob_data, prob_data + num_classes) - prob_data;
          if (curr_label != 0 &&
              (!merge_repeated_ || (previous_label != curr_label))) {
            t_dec++;
            values_cach.push_back(curr_label);
          }
          previous_label = curr_label;
        }
        output_len_data[i] = t_dec;
      }

      int32_t values_cach_size = values_cach.size();
      auto* values =
          Output(VALUES, vector<int64_t>{values_cach_size}, at::dtype<int>());
      int* values_data = values->mutable_data<int>();
      for (size_t i = 0; i < values_cach.size(); ++i) {
        values_data[i] = values_cach.at(i);
      }
      values_cach.clear();

      return true;
        */
    }
}
