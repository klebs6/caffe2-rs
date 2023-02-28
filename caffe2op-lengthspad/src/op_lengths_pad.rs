crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
};

/**
| Given DATA tensor of rank r >= 1, and LENGTHS
| tensor of rank 1, pad each segment in DATA with
| `value`, so that each segment's length is
| `target_length`.
|
| If will throw, if there is segment of length
| larger than `target_length`.
|
| Example:
|   DATA  = [
|       [2.3, 3.4],
|       [4.5, 5.7],
|       [6.8, 7.9],
|   ]
|   LENGTHS = [0, 1, 1, 1]
|   and target_length = 2, padding value = -1.0
|   OUTPUT = [
|     [-1.0, -1.0],
|     [-1.0, -1.0],
|     [2.3, 3.4],
|     [-1.0, -1.0],
|     [4.5, 5.7],
|     [-1.0, -1.0],
|     [6.8, 7.9],
|     [-1.0, -1.0],
|   ]
*/
pub struct LengthsPadOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    padding_value: f64,
    target_length: i32,
    lengths_host:  Tensor, //{CPU};
}

num_inputs!{LengthsPad, 2}

num_outputs!{LengthsPad, 1}

inputs!{LengthsPad, 
    0 => ("DATA",    "Tensor of rank r >= 1. First dimension must be equal to the size of lengths"),
    1 => ("LENGTHS", "Tensor of int32 lengths of rank 1")
}

outputs!{LengthsPad, 
    0 => ("OUTPUT",  "Padded DATA tensor")
}

args!{LengthsPad, 
    0 => ("padding_value", "The value to pad the data"),
    1 => ("target_length", "The target length of each segment")
}

tensor_inference_function!{LengthsPad, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      int target_length = helper.GetSingleArgument<int>("target_length", -1);
      CAFFE_ENFORCE_GE(target_length, 1);
      vector<int> output_dims;
      const auto& data_dims = GetDimsVector(in[0]);
      const auto& lengths_dims = GetDimsVector(in[1]);
      output_dims.push_back(lengths_dims[0] * target_length);
      output_dims.insert(
          output_dims.end(), data_dims.begin() + 1, data_dims.end());

      out[0] = CreateTensorShape(output_dims, in[0].data_type());
      return out;
    } */}


input_tags!{
    LengthsPadOp {
        Data,
        Lengths
    }
}

impl<Context> LengthsPadOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(double, "padding_value", padding_value_, -1),
            OP_SINGLE_ARG(int, "target_length", target_length_, -1) 

        CAFFE_ENFORCE_GE(target_length_, 1, "target_length argument must be >= 1");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double, int32_t, int64_t>>::call(
            this, Input(DATA));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
            auto& lengths = Input(LENGTHS);

            CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be 1-D");
            CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");

            // Context::CopyFrom and math::Sum need the same context to avoid race
            // conditions
            // why? CPUContext is not used in Sum
            lengths_host_.CopyFrom(lengths);

            auto lengths_size = lengths_host_.numel();
            auto* lengths_data = lengths_host_.template data<int32_t>();

            int32_t total_length = 0;
            CPUContext cpuContext;
            math::Sum<int32_t, CPUContext>(
                lengths_size, lengths_data, &total_length, &cpuContext);

            CAFFE_ENFORCE_EQ(total_length, data.size(0));

            auto shape = data.sizes().vec();
            shape[0] = lengths_size * target_length_;
            auto* output = Output(0, shape, at::dtype<T>());

            auto block_size = data.size_from_dim(1);
            auto src_data = data.template data<T>();
            auto out_data = output->template mutable_data<T>();

            math::Set(
                output->numel(), static_cast<T>(padding_value_), out_data, &context_);
            for (int64_t i = 0; i < lengths_size; ++i) {
              auto length = lengths_data[i];
              CAFFE_ENFORCE_GE(length, 0);
              CAFFE_ENFORCE_GE(
                  target_length_,
                  length,
                  "Length at index = ",
                  i,
                  " is larger than target length");

              context_.template CopySameDevice<T>(
                  block_size * length, src_data, out_data);

              out_data += block_size * target_length_;
              src_data += block_size * length;
            }
            return true;
        */
    }
}

register_cpu_operator!{LengthsPad, LengthsPadOp<CPUContext>}

no_gradient!{LengthsPad}
