crate::ix!();

use crate::{
    OperatorStorage,
    Workspace,
    OperatorDef
};


#[inline] pub fn is_little_endian() -> bool {
    
    todo!();
    /*
        constexpr std::int32_t kValue = 1;
      return reinterpret_cast<const std::uint8_t*>(&kValue)[0] == 1;
    */
}

pub type ConvertFnType<T> = fn(
    dst: *mut f32, 
    src: *const T, 
    N:   libc::size_t) -> ();

/**
  | Fake 2/4 bit quantization
  | 
  | Creates a 2/4bit rowwise quantized
  | blob with scales and biases in fp16
  | 
  | The storage format is 8 bit rowwise with
  | scales and biases in fp32
  |
  */
pub struct FloatToFusedNBitFakeRowwiseQuantizedOp<Context,
const BIT_RATE: i32, T, ConvertFn, const GREEDY: bool>
{
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
    phantomCFN: PhantomData<ConvertFn>,
}

input_tags!{
    FloatToFusedNBitFakeRowwiseQuantizedOp {
        DataFloat
    }
}

output_tags!{
    FloatToFusedNBitFakeRowwiseQuantizedOp {
        // INT8 suffix because this is a fake quantization operator whose output
        // type is always 8-bit regardless of BIT_RATE.
        DataFusedScaleBiasInt8
    }
}

impl<Context, const BIT_RATE: i32, T, ConvertFn, const GREEDY: bool>
FloatToFusedNBitFakeRowwiseQuantizedOp<Context, BIT_RATE, T, ConvertFn, GREEDY>
{
    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(internal::is_little_endian(), "Unsupported endianness");

        const auto& input = Input(DATA_FLOAT);

        const auto input_rows = input.size(0);
        const auto input_columns = input.size(1);
        CAFFE_ENFORCE_EQ(input.dim(), 2, "Expect input to be a matrix");

        const std::vector<int64_t> output_dimensions = {input_rows,
                                                        input_columns + 8};
        auto* output = Output(
            DATA_FUSED_SCALE_BIAS_INT8, output_dimensions, at::dtype<uint8_t>());

        const auto* input_data = input.template data<T>();
        auto* output_data = output->template mutable_data<uint8_t>();
        const auto output_columns = output->size(1);

        if (!std::is_same<T, float>::value && !std::is_same<T, at::Half>::value) {
          CAFFE_THROW("Unsupported data type");
        }

        bool use_openmp = GREEDY;
    #ifdef _OPENMP
        vector<float> tmp_vec(input_columns * (GREEDY ? omp_get_max_threads() : 1));
    #else
        vector<float> tmp_vec(input_columns);
    #endif

    #pragma omp parallel for if (GREEDY)
        for (int row = 0; row < input_rows; ++row) {
          float* tmp = tmp_vec.data();
    #ifdef _OPENMP
          if (GREEDY) {
            tmp = &tmp_vec[omp_get_thread_num() * input_columns];
          }
    #endif
          convert(tmp, input_data + row * input_columns, input_columns);
          uint8_t* output_row = output_data + row * output_columns;
          float* output_row_scale_bias =
              reinterpret_cast<float*>(output_row + input_columns);

          float minimum_element = *std::min_element(tmp, tmp + input_columns);
          float maximum_element = *std::max_element(tmp, tmp + input_columns);

          if (GREEDY) {
            internal::param_search_greedy(
                tmp,
                input_columns,
                200,
                0.16,
                minimum_element,
                maximum_element,
                BIT_RATE);
          }

          minimum_element = static_cast<at::Half>(minimum_element);
          const float range = maximum_element - minimum_element;

          const float scale = range == 0
              ? 1.0f
              : static_cast<float>(static_cast<at::Half>(
                    range / static_cast<float>((1 << BIT_RATE) - 1)));
          const float inverse_scale = 1.0f / scale;

          output_row_scale_bias[0] = scale;
          output_row_scale_bias[1] = minimum_element;

          for (size_t col = 0; col < input_columns; ++col) {
            output_row[col] = std::max(
                0,
                std::min<int>(
                    std::lrintf((tmp[col] - minimum_element) * inverse_scale),
                    (1 << BIT_RATE) - 1));
          }
        }

        return true;
        */
    }
}

#[inline] pub fn compress_uniform_simplified_(
    x:         *const f32,
    n:         i32,
    xmin:      f32,
    xmax:      f32,
    xq:        *mut f32,
    bit_rate:  i32) -> f32 
{
    todo!();
    /*
        xmin = static_cast<at::Half>(xmin);
      float data_range = xmax - xmin;
      float qmax = (1 << bit_rate) - 1;
      float scale = data_range == 0
          ? 1.0
          : static_cast<float>(static_cast<at::Half>(data_range / qmax));
      float inverse_scale = 1.0f / scale;

      float norm = 0.0f;
      constexpr int VLEN = 8;
      int i = 0;

    #ifdef __AVX__
      // vectorized loop
      __m256 norm_v = _mm256_setzero_ps();
      for (; i < N / VLEN * VLEN; i += VLEN) {
        __m256 X_v = _mm256_loadu_ps(X + i);
        // Affine
        __m256 Xq_v = _mm256_mul_ps(
            _mm256_sub_ps(X_v, _mm256_set1_ps(xmin)),
            _mm256_set1_ps(inverse_scale));
        // Round
        // Use _MM_FROUND_CUR_DIRECTION to match the behavior with the remainder
        // code. In most cases, the rounding mode is round-to-nearest-even.
        Xq_v = _mm256_round_ps(Xq_v, _MM_FROUND_CUR_DIRECTION);
        // Clip
        Xq_v = _mm256_max_ps(
            _mm256_setzero_ps(), _mm256_min_ps(Xq_v, _mm256_set1_ps(qmax)));
        // Inverse affine
        Xq_v = _mm256_add_ps(
            _mm256_mul_ps(Xq_v, _mm256_set1_ps(scale)), _mm256_set1_ps(xmin));
        __m256 err_v = _mm256_sub_ps(X_v, Xq_v);
        norm_v = _mm256_add_ps(_mm256_mul_ps(err_v, err_v), norm_v);
      }
      alignas(64) float temp[VLEN];
      _mm256_store_ps(temp, norm_v);
      for (int j = 0; j < VLEN; ++j) {
        norm += temp[j];
      }
    #endif // __AVX__

      // remainder loop
      for (; i < N; i++) {
        Xq[i] = std::max(
            0.0f, std::min<float>(nearbyint((X[i] - xmin) * inverse_scale), qmax));
        Xq[i] = Xq[i] * scale + xmin;
        norm += (X[i] - Xq[i]) * (X[i] - Xq[i]);
      }

      return std::sqrt(norm);
    */
}

#[inline] pub fn convertfp_32fp32(dst: *mut f32, src: *const f32, n: usize)  {
    
    todo!();
    /*
        memcpy(dst, src, sizeof(float) * N);
    */
}

#[inline] pub fn convertfp_16fp32(dst: *mut f32, src: *const f16, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}

/**
  | @params Xmin initial solution passed
  | and potentiall better solution returns
  | 
  | @params Xmax initial solution passed
  | and potentiall better solution returns
  |
  */
#[inline] pub fn param_search_greedy(
    x:          *const f32,
    n:          i32,
    n_bins:     Option<i32>,
    ratio:      Option<f32>,
    xmin:       &mut f32,
    xmax:       &mut f32,
    bit_rate:   i32)  
{
    let n_bins: i32 = n_bins.unwrap_or(200);
    let ratio: f32 = ratio.unwrap_or(0.16);

    todo!();
    /*
        float stepsize = (Xmax - Xmin) / n_bins;
      int min_bins = n_bins * (1 - ratio);

      vector<float> Xq(N);

      float loss =
          compress_uniform_simplified_(X, N, Xmin, Xmax, Xq.data(), bit_rate);
      float best_loss = loss;

      float cur_min = Xmin;
      float cur_max = Xmax;
      float cur_loss = loss;

      float thr = min_bins * stepsize;
      while (cur_min + thr < cur_max) {
        // move left
        float loss1 = compress_uniform_simplified_(
            X, N, cur_min + stepsize, cur_max, Xq.data(), bit_rate);
        // move right
        float loss2 = compress_uniform_simplified_(
            X, N, cur_min, cur_max - stepsize, Xq.data(), bit_rate);
        if (cur_loss < loss1 && cur_loss < loss2 && cur_loss < best_loss) {
          // found a local optima
          best_loss = cur_loss;
          Xmin = cur_min;
          Xmax = cur_max;
        }
        if (loss1 < loss2) {
          cur_min = cur_min + stepsize;
          cur_loss = loss1;
        } else {
          cur_max = cur_max - stepsize;
          cur_loss = loss2;
        }
      }
    */
}

/**
  | Applies 4-bit row-wise fake quantization
  | to a tensor of floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    FloatToFused4BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f32,
        internal::convertfp32fp32>
}

register_cpu_operator_with_engine!{
    FloatToFused4BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f32,
        internal::convertfp32fp32,
        true /* GREEDY */>
}

no_gradient!{FloatToFused4BitFakeRowwiseQuantized}

num_inputs!{FloatToFused4BitFakeRowwiseQuantized, 1}

num_outputs!{FloatToFused4BitFakeRowwiseQuantized, 1}

inputs!{FloatToFused4BitFakeRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused4BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused4BitFakeRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

/**
  | Applies 4-bit row-wise fake quantization
  | to a tensor of half floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    HalfToFused4BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f16,
        internal::convertfp16fp32>
}

register_cpu_operator_with_engine!{
    HalfToFused4BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f16,
        internal::convertfp16fp32,
        true /* GREEDY */>
}

no_gradient!{HalfToFused4BitFakeRowwiseQuantized}

num_inputs!{HalfToFused4BitFakeRowwiseQuantized, 1}

num_outputs!{HalfToFused4BitFakeRowwiseQuantized, 1}

inputs!{HalfToFused4BitFakeRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfToFused4BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfToFused4BitFakeRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

/**
  | Applies 2-bit row-wise fake quantization
  | to a tensor of floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    FloatToFused2BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f32,
        internal::convertfp32fp32>
}

no_gradient!{FloatToFused2BitFakeRowwiseQuantized}

register_cpu_operator_with_engine!{
    FloatToFused2BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f32,
        internal::convertfp32fp32,
        true /* GREEDY */>
}

num_inputs!{FloatToFused2BitFakeRowwiseQuantized, 1}

num_outputs!{FloatToFused2BitFakeRowwiseQuantized, 1}

inputs!{FloatToFused2BitFakeRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused2BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused2BitFakeRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(1, X.dims(1) + 8);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}

/**
  | Applies 2-bit row-wise fake quantization
  | to a tensor of half floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    HalfToFused2BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f16,
        convertfp16fp32> 
}

no_gradient!{HalfToFused2BitFakeRowwiseQuantized}

register_cpu_operator_with_engine!{
    HalfToFused2BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f16,
        convertfp16fp32,
        Greedy>
}

num_inputs!{HalfToFused2BitFakeRowwiseQuantized, 1}

num_outputs!{HalfToFused2BitFakeRowwiseQuantized, 1}

inputs!{HalfToFused2BitFakeRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfToFused2BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfToFused2BitFakeRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(1, X.dims(1) + 8);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}
