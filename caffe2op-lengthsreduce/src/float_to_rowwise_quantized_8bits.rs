crate::ix!();

pub const kEqualityThreshold: f32 = 1e-10;

/**
  | This operator applies 8Bit row-wise
  | quantization to input tensor and returns
  | quantized tensor.
  | 
  | Row wise quantization of input tensor
  | is the following process.
  | 
  | We take tensor of size
  | 
  | (m_1, m_2,...,m_n), n >= 2, reshape
  | it into matrix of size
  | 
  | (m_1, m_2 x... x m_n) and apply row-wise
  | quantization.
  | 
  | After this, we compute scale_i= (min_i
  | - max_i) / 255 and bias_i = min_i for i-th
  | row r_i of reshaped matrix, where min_i
  | and max_i -- minimum and maximum elements
  | of i-th row, and quantize each element
  | r_{ij} as 0 <= round(r_ij - bias_i) /
  | scale_i) < 256.
  | 
  | Instead of input tensor we obtain uint8
  | tensor and auxiliary information as
  | scale and bias to restore input tensor
  | (with losses).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FloatToRowwiseQuantized8BitsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

no_gradient!{FloatToRowwiseQuantized8Bits}

register_cpu_operator!{
    FloatToRowwiseQuantized8Bits, 
    FloatToRowwiseQuantized8BitsOp<CPUContext>
}

num_inputs!{FloatToRowwiseQuantized8Bits, 1}

num_outputs!{FloatToRowwiseQuantized8Bits, 2}

inputs!{FloatToRowwiseQuantized8Bits, 
    0 => ("input", "input")
}

outputs!{FloatToRowwiseQuantized8Bits, 
    0 => ("quantized_input", "quantized_input"),
    1 => ("scale_bias",      "Matrix of floats, each row r_i of which stores a pair s_i, b_i")
}

input_tags!{
    FloatToRowwiseQuantized8BitsOp {
        DataFloat
    }
}

output_tags!{
    FloatToRowwiseQuantized8BitsOp {
        DataUint8,
        ScaleBias
    }
}

impl<Context> FloatToRowwiseQuantized8BitsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(DATA_FLOAT);

        auto* input_data = input.template data<float>();
        auto* output = Output(DATA_UINT8, input.sizes(), at::dtype<uint8_t>());
        vector<int64_t> scale_bias_dims = {input.size(0), 2};
        auto* scale_bias = Output(SCALE_BIAS, scale_bias_dims, at::dtype<float>());
        auto* output_data = output->template mutable_data<uint8_t>();
        float* scale_bias_data = scale_bias->template mutable_data<float>();
        size_t n_blocks = input.size(0);
        size_t block_size = input.size_from_dim(1);
        for (size_t i = 0; i < n_blocks; ++i) {
          ConstEigenVectorArrayMap<float> input_row(
              input_data + i * block_size, block_size);
          EigenVectorArrayMap<uint8_t> output_row(
              output_data + i * block_size, block_size);
          auto min_element = input_row.minCoeff();
          auto max_element = input_row.maxCoeff();
          if (max_element - min_element < kEqualityThreshold) {
            scale_bias_data[2 * i] = 1.0f;
            scale_bias_data[2 * i + 1] = min_element;
            memset(output_data + i * block_size, 0, block_size);
          } else {
            scale_bias_data[2 * i] = (max_element - min_element) / 255.0f;
            scale_bias_data[2 * i + 1] = min_element;
            const float inv_scale = 1.0f / scale_bias_data[2 * i];
            output_row = ((input_row - scale_bias_data[2 * i + 1]) * inv_scale)
                             .round()
                             .template cast<uint8_t>();
          }
        }
        return true;
        */
    }
}
