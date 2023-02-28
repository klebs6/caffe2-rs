crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    Tensor,
    OperatorDef,
    TensorShape,
    Workspace,
    CPUContext,
};

///------------------------------------
pub struct FloatToHalfOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    clip: bool,
}

num_inputs!{FloatToHalf, 1}

num_outputs!{FloatToHalf, 1}

tensor_inference_function!{FloatToHalf, /* [](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      const TensorShape& X = in[0];
      out.push_back(X);
      out[0].set_data_type(TensorProto_DataType_FLOAT16);

      return out;
    } */
}

impl<Context> FloatToHalfOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            clip_(this->template GetSingleArgument<bool>("clip", false))
        */
    }
}

///-------------------------
pub struct HalfToFloatOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{HalfToFloat, 1}

num_outputs!{HalfToFloat, 1}

tensor_inference_function!{HalfToFloat, /* [](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      const TensorShape& X = in[0];
      out.push_back(X);
      out[0].set_data_type(TensorProto_DataType_FLOAT);

      return out;
    } */
}

///-----------------
pub struct Float16ConstantFillOp {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage: OperatorStorage,
    context: CPUContext,

    shape: Vec<i64>,
}

num_inputs!{Float16ConstantFill, 0}

num_outputs!{Float16ConstantFill, 1}

outputs!{Float16ConstantFill, 
    0 => ("output", "Output tensor of constant values specified by 'value'")
}

args!{Float16ConstantFill, 
    0 => ("value", "The value for the elements of the output tensor."),
    1 => ("shape", "The shape of the output tensor.")
}

tensor_inference_function!{Float16ConstantFill, Float16FillerTensorInference }

impl Float16ConstantFillOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            shape_(this->template GetRepeatedArgument<int64_t>("shape"))
        */
    }
}

/**
  | Fills a half float tensor of a specified
  | shape with values from a uniform distribution[min,max]
  |
  */
pub struct Float16UniformFillOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:          OperatorStorage,
    context:          Context,

    shape:            Vec<i64>,
    min:              f32,
    max:              f32,
    temp_data_buffer: Tensor,
}

num_inputs!{Float16UniformFill, 0}

num_outputs!{Float16UniformFill, 1}

args!{Float16UniformFill, 
    0 => ("shape", "Shape of the tensor"),
    1 => ("min", "Minimim value to generate"),
    2 => ("max", "Maximum value to generate")
}

tensor_inference_function!{Float16UniformFill, /* Float16FillerTensorInference */}

impl<Context> Float16UniformFillOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            shape_(this->template GetRepeatedArgument<int64_t>("shape")),
            min_(this->template GetSingleArgument<float>("min", 0)),
            max_(this->template GetSingleArgument<float>("max", 1)) 

        if (InputSize() == 3) {
          CAFFE_ENFORCE(
              !this->template HasSingleArgumentOfType<float>("min"),
              "Cannot set both min arg and min input blob");
          CAFFE_ENFORCE(
              !this->template HasSingleArgumentOfType<float>("max"),
              "Cannot set both max arg and max input blob");
        } else {
          CAFFE_ENFORCE_LT(
              min_, max_, "Max value should be bigger than min value.");
        }
        */
    }
}

#[inline] pub fn float16_filler_tensor_inference(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      out[0].set_data_type(static_cast<TensorProto_DataType>(
          helper.GetSingleArgument<int>("dtype", TensorProto_DataType_FLOAT16)));
      auto shape = helper.GetRepeatedArgument<int>("shape");
      for (int d : shape) {
        out[0].add_dims(d);
      }
      return out;
    */
}

#[inline] pub fn float_to_float16_ref(
    input:   *const f32,
    out:     *mut f16,
    n:       usize,
    do_clip: Option<bool>)  
{
    let do_clip: bool = do_clip.unwrap_or(false);

    todo!();
    /*
        if (do_clip) {
        constexpr float FP16_MAX = 65504.f;
        for (size_t i = 0; i < N; ++i) {
          out[i] = std::max(-FP16_MAX, std::min(in[i], FP16_MAX));
        }
      } else {
        for (size_t i = 0; i < N; ++i) {
          out[i] = in[i];
        }
      }
    */
}

#[inline] pub fn float_16to_float_ref(input: *const f16, out: *mut f32, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; ++i) {
        out[i] = in[i];
      }
    */
}

impl FloatToHalfOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

      auto* output = Output(0, input.sizes(), at::dtype<at::Half>());
      const float* data = input.template data<float>();
      at::Half* out = output->template mutable_data<at::Half>();
      auto N = input.numel();

    #ifdef USE_FBGEMM
      // There exists a verion fbgemm::FloatToFloat16_simd which will issue avx-512
      // instructions when possible. However, this actually doesn't give perf
      // benefits, according to benchmarks on T1/T6. Hence we stick to avx2 versions
      // here.
      if (GetCpuId().avx2()) {
        fbgemm::FloatToFloat16_avx2(
            data, reinterpret_cast<fbgemm::float16*>(out), N, clip_);
      } else {
        FloatToFloat16_ref(data, out, N, clip_);
      }
    #else
      FloatToFloat16_ref(data, out, N, clip_);
    #endif

      return true;
        */
    }
}

impl HalfToFloatOp<CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

      auto* output = Output(0, input.sizes(), at::dtype<float>());
      const at::Half* data = input.template data<at::Half>();
      float* out = output->template mutable_data<float>();
      auto N = input.numel();

    #ifdef USE_FBGEMM
      // Same reasoning of sticking to avx2
      if (GetCpuId().avx2()) {
        fbgemm::Float16ToFloat_avx2(
            reinterpret_cast<const fbgemm::float16*>(data), out, N);
      } else {
        Float16ToFloat_ref(data, out, N);
      }
    #else
      Float16ToFloat_ref(data, out, N);
    #endif

      return true;
        */
    }
}

register_cpu_operator!{FloatToHalf, FloatToHalfOp<CPUContext>}

register_cpu_operator!{HalfToFloat, HalfToFloatOp<CPUContext>}

impl Float16ConstantFillOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(0, shape_, at::dtype<at::Half>());
      const float givenValue =
          this->template GetSingleArgument<float>("value", 0.0f);
      at::Half givenFp16Value = givenValue;

      if (output->numel()) {
        at::Half* out = output->template mutable_data<at::Half>();
        std::fill(out, out + output->numel(), givenFp16Value);
      }
      return true;
        */
    }
}

impl Float16UniformFillOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(0, shape_, at::dtype<at::Half>());
      at::Half* out = output->template mutable_data<at::Half>();

      // Get a batch row by row and convert
      auto leading_dim_sz = output->size(0);
      int rowsz = output->numel() / output->size(0);

      vector<float> intermediate_data_;
      intermediate_data_.resize(rowsz);
      for (uint64_t i = 0; i < leading_dim_sz; i++) {
        math::RandUniform<float, CPUContext>(
            rowsz, min_, max_, intermediate_data_.data(), &context_);
        for (uint64_t j = 0; j < rowsz; j++) {
          out[i * rowsz + j] = intermediate_data_[j];
        }
      }
      return true;
        */
    }
}

register_cpu_operator!{Float16ConstantFill, Float16ConstantFillOp}

register_cpu_operator!{Float16UniformFill,  Float16UniformFillOp<CPUContext>}

no_gradient!{Float16UniformFill}

pub struct GetFloatToHalfGradient ;

impl GetGradientDefs for GetFloatToHalfGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "HalfToFloat", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{FloatToHalf, GetFloatToHalfGradient}

pub struct GetHalfToFloatGradient ;

impl GetGradientDefs for GetHalfToFloatGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "FloatToHalf", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }

}

register_gradient!{HalfToFloat, GetHalfToFloatGradient}

no_gradient!{Float16ConstantFill}
