crate::ix!();

/**
  | Given uint8 tensor, quantized using
  | 8bit row-wise quantization, and auxiliary
  | scales and biases, this operator restores
  | float tensor in the following way.
  | 
  | We take input 8bits tensor of size
  | 
  | (m_1, m_2, ..., m_n), n >= 2, reshape
  | it into matrix of size
  | 
  | (m_1, m_2 x... x m_n).
  | 
  | We compute element r_{ij} of output
  | matrix as
  | 
  | r_{ij} * s_i + b_i
  | 
  | and after this we reshape this output
  | matrix into output tensor of size
  | 
  | (m_1, m_2, ..., m_n).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct Rowwise8BitQuantizedToFloatOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    Rowwise8BitQuantizedToFloat, 
    Rowwise8BitQuantizedToFloatOp<CPUContext>
}

num_inputs!{Rowwise8BitQuantizedToFloat, 2}

num_outputs!{Rowwise8BitQuantizedToFloat, 1}

inputs!{Rowwise8BitQuantizedToFloat, 
    0 => ("quantized_input",  "quantized_input"),
    1 => ("scale_bias",       "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{Rowwise8BitQuantizedToFloat, 
    0 => ("output", "output")
}

no_gradient!{Rowwise8BitQuantizedToFloat}

input_tags!{
    Rowwise8BitQuantizedToFloatOp {
        DataUint8,
        ScaleBias
    }
}

output_tags!{
    Rowwise8BitQuantizedToFloatOp {
        DataFloat
    }
}

impl<Context> Rowwise8BitQuantizedToFloatOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(DATA_UINT8);
        auto& scale_bias = Input(SCALE_BIAS);

        CAFFE_ENFORCE_EQ(2, scale_bias.dim(), "scale_bias has to be matrix");
        CAFFE_ENFORCE_EQ(
            input.size(0),
            scale_bias.size(0),
            "scale_bias must have the same first dim as data");
        CAFFE_ENFORCE_EQ(
            2,
            scale_bias.size(1),
            "the second dim of scale_bias has to be equal to 2");
        auto* output = Output(DATA_FLOAT, input.sizes(), at::dtype<float>());
        auto* input_data = input.template data<uint8_t>();
        auto* scale_bias_data = scale_bias.template data<float>();

        auto* output_data = output->template mutable_data<float>();
        size_t block_size = input.size_from_dim(1);
        size_t n_blocks = input.size(0);

        for (size_t i = 0; i < n_blocks; ++i) {
          ConstEigenVectorArrayMap<uint8_t> input_row(
              input_data + i * block_size, block_size);
          EigenVectorArrayMap<float> output_row(
              output_data + i * block_size, block_size);
          output_row = input_row.template cast<float>() * scale_bias_data[2 * i] +
              scale_bias_data[2 * i + 1];
        }
        return true;
        */
    }
}

