crate::ix!();

/**
  | Fills a half float tensor of a specified
  | shape with values from a uniform distribution[min,max]
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct Float16UniformFillOp<Context> {
    storage:          OperatorStorage,
    context:          Context,
    shape:            Vec<i64>,
    min:              f32,
    max:              f32,
    temp_data_buffer: Tensor,
}

register_cpu_operator!{Float16UniformFill,  Float16UniformFillOp<CPUContext>}

no_gradient!{Float16UniformFill}

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
