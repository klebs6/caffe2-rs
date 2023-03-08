crate::ix!();

use crate::{
    OperatorStorage,
};

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

///---------------------------

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseLengths8BitsRowwiseOp<Context, const USE_WEIGHTS: bool, const USE_MEAN: bool, OutDataT> {

    storage:         OperatorStorage,
    context:         Context,

    phantomOutDataT: PhantomData<OutDataT>,
}

//OutDataT default type is f32
impl<Context, 
    const USE_WEIGHTS: bool, 
    const USE_MEAN: bool, 
    OutDataT> 
SparseLengths8BitsRowwiseOp<Context, USE_WEIGHTS, USE_MEAN, OutDataT> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self, ) -> bool {
        todo!();
        /*
            auto& dataInput = Input(DATA);
        auto& lengthsInput = Input(LENGTHS);

        auto* scale_bias = Input(SCALE_BIAS).template data<float>();
        CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
        const int64_t outputSize = lengthsInput.size(0);

        auto& indicesInput = Input(INDICES);
        CAFFE_ENFORCE_EQ(2, Input(SCALE_BIAS).dim(), "scale_bias has to be matrix");
        CAFFE_ENFORCE_EQ(
            dataInput.size(0),
            Input(SCALE_BIAS).size(0),
            "scale_bias must have the same first dim as data");
        CAFFE_ENFORCE_EQ(
            2,
            Input(SCALE_BIAS).size(1),
            "the second dim of scale_bias has to be equal to 2");
        CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
        const IndexType* indices = indicesInput.template data<IndexType>();
        int64_t dataToReduceSize = indicesInput.size(0);

        const int* lengths = lengthsInput.template data<int>();
        vector<int64_t> shape = dataInput.sizes().vec();
        shape[0] = outputSize;
        auto* output = Output(0, shape, at::dtype<OutDataT>());
        const float* w = nullptr;
        if (USE_WEIGHTS) {
          w = Input(WEIGHTS).template data<float>();
        }
        int64_t in_block_size = dataInput.size_from_dim(1);
        OutDataT* out = output->template mutable_data<OutDataT>();
        const uint8_t* input_data = dataInput.template data<uint8_t>();

        // delegate work to perfkernel that branches based on architecture
        const int64_t indices_size = indicesInput.numel();
        const int64_t N = dataInput.size(0);
        EmbeddingLookup(
            in_block_size,
            outputSize,
            indices_size,
            N, // embedding table length
            input_data,
            indices,
            lengths,
            w,
            scale_bias,
            USE_MEAN,
            out);

        return true;
        */
    }
}

/**
  |there is weirdness with MaybeWeights it is
  |possible Data is 0, MaybeWeights is 1, Indices
  |is 2, etc or it is possible Data is 0, Indices
  |is 1, etc check the c++ code
  */
pub enum SparseLengths8BitsRowwiseOpIdx {
    Data,
    MaybeWeights,
    Indices,
    Lengths,
    ScaleBias,
}

/**
  | Variation of SparseLengthsSum operator,
  | where DATA is stored using 8bits.
  | 
  | DATA was quantized with 8Bit row-wise
  | quantization (see doc to FloatToRowwiseQuantized8Bits
  | operator).
  | 
  | To restore DATA from 8Bit, we use additional
  | input that stores scales and biases.
  |
  */
register_cpu_operator!{SparseLengthsSum8BitsRowwise, SparseLengths8BitsRowwiseOp<CPUContext>}

num_inputs!{SparseLengthsSum8BitsRowwise, 4}

num_outputs!{SparseLengthsSum8BitsRowwise, 1}

inputs!{SparseLengthsSum8BitsRowwise, 
    0 => ("DATA",         "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("INDICES",      "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS",      "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("scale_bias",   "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{SparseLengthsSum8BitsRowwise, 
    0 => ("output", "output")
}

no_gradient!{SparseLengthsSum8BitsRowwise}

/**
  | Variation of SparseLengthsWeightedSum
  | operator, where
  | 
  | DATA is stored using 8bits. DATA was
  | quantized with 8Bit row-wise quantization
  | (see doc to FloatToRowwiseQuantized8Bits
  | operator). To restore DATA from 8Bit,
  | we use additional input that stores
  | scales and biases.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum8BitsRowwise, 
    SparseLengths8BitsRowwiseOp<CPUContext, 1>
}

num_inputs!{SparseLengthsWeightedSum8BitsRowwise, 5}

num_outputs!{SparseLengthsWeightedSum8BitsRowwise, 1}

inputs!{SparseLengthsWeightedSum8BitsRowwise, 
    0 => ("DATA",          "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("SCALARS",       "Scalar multipliers for the input slices. Must be a vector with the length matching the length of INDICES"),
    2 => ("INDICES",       "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS",       "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("scale_bias",    "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{SparseLengthsWeightedSum8BitsRowwise, 
    0 => ("output", "output")
}

no_gradient!{SparseLengthsWeightedSum8BitsRowwise}

/**
  | Variation of SparseLengthsMean operator,
  | where DATA is stored using 8bits.
  | 
  | DATA was quantized with 8Bit row-wise
  | quantization (see doc to FloatToRowwiseQuantized8Bits
  | operator).
  | 
  | To restore DATA from 8Bit, we use additional
  | input that stores scales and biases.
  |
  */
register_cpu_operator!{SparseLengthsMean8BitsRowwise, SparseLengths8BitsRowwiseOp<CPUContext, 0, 1>}

num_inputs!{SparseLengthsMean8BitsRowwise, 4}

num_outputs!{SparseLengthsMean8BitsRowwise, 1}

inputs!{SparseLengthsMean8BitsRowwise, 
    0 => ("DATA",          "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("INDICES",       "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS",       "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("scale_bias",    "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{SparseLengthsMean8BitsRowwise, 
    0 => ("output", "output")
}

no_gradient!{SparseLengthsMean8BitsRowwise}

/**
  | Variation of SparseLengthsWeightedMean
  | operator, where
  | 
  | DATA is stored using 8bits.
  | 
  | DATA was quantized with 8Bit row-wise
  | quantization (see doc to FloatToRowwiseQuantized8Bits
  | operator).
  | 
  | To restore DATA from 8Bit, we use additional
  | input that stores scales and biases.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedMean8BitsRowwise, 
    SparseLengths8BitsRowwiseOp<CPUContext, 1, 1>
}

num_inputs!{SparseLengthsWeightedMean8BitsRowwise, 5}

num_outputs!{SparseLengthsWeightedMean8BitsRowwise, 1}

inputs!{SparseLengthsWeightedMean8BitsRowwise, 
    0 => ("DATA",         "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("SCALARS",      "Scalar multipliers for the input slices. Must be a vector with the length matching the length of INDICES"),
    2 => ("INDICES",      "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS",      "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("scale_bias",   "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{SparseLengthsWeightedMean8BitsRowwise, 
    0 => ("output", "output")
}

no_gradient!{SparseLengthsWeightedMean8BitsRowwise}

