crate::ix!();

/**
  | ForEach is a unary functor that forwards
  | each element of the input array into
  | the elementwise Functor provided,
  | and gathers the results of each call
  | into the resulting array.
  | 
  | Use it as an adaptor if you want to create
  | a UnaryElementwiseOp that acts on each
  | element of the tensor per function call
  | -- this is reasonable for complex types
  | where vectorization wouldn't be much
  | of a gain, performance-wise.
  |
  */
pub struct ForEach<Functor> {
    functor:  Functor,
}

impl<Functor> ForEach<Functor> {

    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : functor(op)
        */
    }
    
    #[inline] pub fn invoke<In, Out, Context>(&mut self, 
        n:       i32,
        input:   *const In,
        out:     *mut Out,
        context: *mut Context) -> bool {
    
        todo!();
        /*
            for (int i = 0; i < n; ++i) {
          out[i] = functor(in[i]);
        }
        return true;
        */
    }
}

pub type StringElementwiseOp<
ScalarFunctor, 
    TypeMap = FixedType<String>>
    = UnaryElementwiseWithArgsOp<
    TensorTypes<String>,
    CPUContext,
    ForEach<ScalarFunctor>,
    TypeMap>;

///-----------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct StringJoinOp<Context> {
    storage: OperatorStorage,
    context: Context,

    delimiter:  String,
    axis:       i32,
}

impl<Context> StringJoinOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            delimiter_( this->template GetSingleArgument<std::string>("delimiter", ",")),
            axis_(this->template GetSingleArgument<int>("axis", 0)) 

        CAFFE_ENFORCE(axis_ == 0 || axis_ == 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<
            float,
            double,
            int8_t,
            uint8_t,
            int16_t,
            uint16_t,
            int32_t,
            int64_t,
            std::string,
            bool>>::call(this, Input(0));
        */
    }
}

impl StringJoinOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& input = Input(0);

      CAFFE_ENFORCE_GT(input.numel(), 0);
      CAFFE_ENFORCE_LE(input.dim(), 2, "Only 1-D and 2-D tensors are supported");

      const auto* inputData = input.data<T>();
      int rowSize = (input.dim() == 2) ? input.size(1) : 1;
      if (this->axis_ == 0) {
        auto* output = Output(0, {input.size(0)}, at::dtype<std::string>());
        auto* outputData = output->template mutable_data<std::string>();

        int offset = 0;
        for (int i = 0; i < input.size(0); ++i) {
          std::stringstream stream;
          std::copy(
              inputData + offset,
              inputData + offset + rowSize,
              std::ostream_iterator<T>(stream, delimiter_.c_str()));
          outputData[i] = stream.str();
          offset += rowSize;
        }
      } else if (this->axis_ == 1) {
        auto* output = Output(0, {input.size(1)}, at::dtype<std::string>());
        auto* outputData = output->template mutable_data<std::string>();

        for (int j = 0; j < input.size(1); ++j) {
          std::stringstream stream;
          for (int i = 0; i < input.size(0); ++i) {
            stream << inputData[i * rowSize + j] << delimiter_;
          }
          outputData[j] = stream.str();
        }
      } else {
        CAFFE_ENFORCE(false, "Not supported");
      }

      return true;
        */
    }
}


pub struct StartsWith {
    prefix:  String,
}

impl StartsWith {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : prefix_(op.GetSingleArgument<std::string>("prefix", ""))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> bool {
        
        todo!();
        /*
            return std::mismatch(prefix_.begin(), prefix_.end(), str.begin()).first ==
            prefix_.end();
        */
    }
}

///--------------------
struct EndsWith {
    suffix:  String,
}

impl EndsWith {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : suffix_(op.GetSingleArgument<std::string>("suffix", ""))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> bool {
        
        todo!();
        /*
            return std::mismatch(suffix_.rbegin(), suffix_.rend(), str.rbegin())
                   .first == suffix_.rend();
        */
    }
}

///---------------------------
pub struct StrEquals {

    text:  String,

}

impl StrEquals {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : text_(op.GetSingleArgument<std::string>("text", ""))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> bool {
        
        todo!();
        /*
            return str == text_;
        */
    }
}

///----------------------------
pub struct Prefix {

    length:  i32,

}

impl Prefix {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : length_(op.GetSingleArgument<int>("length", 3))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> String {
        
        todo!();
        /*
            return std::string(str.begin(), std::min(str.end(), str.begin() + length_));
        */
    }
}

///--------------------------
pub struct Suffix {
    length: i32,
}

impl Suffix {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : length_(op.GetSingleArgument<int>("length", 3))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> String {
        
        todo!();
        /*
            return std::string(std::max(str.begin(), str.end() - length_), str.end());
        */
    }
}

/**
  | Computes the element-wise string prefix
  | of the string tensor.
  | 
  | Input strings that are shorter than
  | prefix length will be returned unchanged.
  | 
  | -----------
  | @note
  | 
  | Prefix is computed on number of bytes,
  | which may lead to wrong behavior and
  | potentially invalid strings for variable-length
  | encodings such as utf-8.
  |
  */
register_cpu_operator!{StringPrefix, StringElementwiseOp<Prefix>}

should_not_do_gradient!{StringPrefix}

num_inputs!{StringPrefix, 1}

num_outputs!{StringPrefix, 1}

inputs!{StringPrefix, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringPrefix, 
    0 => ("prefixes", "Tensor of std::string containing prefixes for each input.")
}

args!{StringPrefix, 
    0 => ("length", "Maximum size of the prefix, in bytes.")
}

/**
  | Computes the element-wise string suffix
  | of the string tensor.
  | 
  | Input strings that are shorter than
  | suffix length will be returned unchanged.
  | 
  | -----------
  | @note
  | 
  | Prefix is computed on number of bytes,
  | which may lead to wrong behavior and
  | potentially invalid strings for variable-length
  | encodings such as utf-8.
  |
  */
register_cpu_operator!{StringSuffix, StringElementwiseOp<Suffix>}

should_not_do_gradient!{StringSuffix}

num_inputs!{StringSuffix, 1}

num_outputs!{StringSuffix, 1}

inputs!{StringSuffix, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringSuffix, 
    0 => ("suffixes", "Tensor of std::string containing suffixes for each output.")
}

args!{StringSuffix, 
    0 => ("length", "Maximum size of the suffix, in bytes.")
}

/**
  | Performs the starts-with check on each
  | string in the input tensor.
  | 
  | Returns tensor of boolean of the same
  | dimension of input.
  |
  */
register_cpu_operator!{StringStartsWith,
    StringElementwiseOp<StartsWith, FixedType<bool>>}

should_not_do_gradient!{StringStartsWith}

num_inputs!{StringStartsWith, 1}

num_outputs!{StringStartsWith, 1}

inputs!{StringStartsWith, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringStartsWith, 
    0 => ("bools", "Tensor of bools of same shape as input.")
}

args!{StringStartsWith, 
    0 => ("prefix", "The prefix to check input strings against.")
}

/**
  | Performs the ends-with check on each
  | string in the input tensor.
  | 
  | Returns tensor of boolean of the same
  | dimension of input.
  |
  */
register_cpu_operator!{StringEndsWith,
    StringElementwiseOp<EndsWith, FixedType<bool>>}

num_inputs!{StringEndsWith, 1}

num_outputs!{StringEndsWith, 1}

inputs!{StringEndsWith, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringEndsWith, 
    0 => ("bools", "Tensor of bools of same shape as input.")
}

args!{StringEndsWith, 
    0 => ("suffix", "The suffix to check input strings against.")
}

should_not_do_gradient!{StringEndsWith}

/**
  | Performs equality check on each string
  | in the input tensor.
  | 
  | Returns tensor of booleans of the same
  | dimension as input.
  |
  */
register_cpu_operator!{StringEquals,
    StringElementwiseOp<StrEquals, FixedType<bool>>}

num_inputs!{StringEquals, 1}

num_outputs!{StringEquals, 1}

inputs!{StringEquals, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringEquals, 
    0 => ("bools", "Tensor of bools of same shape as input.")
}

args!{StringEquals, 
    0 => ("text", "The text to check input strings equality against.")
}

should_not_do_gradient!{StringEquals}

/**
  | Takes a 1-D or a 2-D tensor as input and
  | joins elements in each row with the provided
  | delimiter.
  | 
  | Output is a 1-D tensor of size equal to
  | the first dimension of the input.
  | 
  | Each element in the output tensor is
  | a string of concatenated elements corresponding
  | to each row in the input tensor.
  | 
  | For 1-D input, each element is treated
  | as a row.
  |
  */
register_cpu_operator!{StringJoin, StringJoinOp<CPUContext>}

num_inputs!{StringJoin, 1}

num_outputs!{StringJoin, 1}

inputs!{StringJoin, 
    0 => ("input", "1-D or 2-D tensor")
}

outputs!{StringJoin, 
    0 => ("strings", "1-D tensor of strings created by joining row elements from the input tensor.")
}

args!{StringJoin, 
    0 => ("delimiter", "Delimiter for join (Default: ,)."),
    1 => ("axis", "Axis for the join (either 0 or 1)")
}

should_not_do_gradient!{StringJoin}
