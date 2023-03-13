crate::ix!();

/**
  | Similar as LengthsToSegmentIds but
  | output vector of segment weights derived
  | by lengths. i.e 1/pow(length, power)
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToWeightsOp<Context> {
    storage: OperatorStorage,
    context: Context,
    power:   f32,
}

num_inputs!{LengthsToWeights, 1}

num_outputs!{LengthsToWeights, 1}

inputs!{LengthsToWeights, 
    0 => ("lengths", "1-D int32_t or int64_t tensor of lengths")
}

outputs!{LengthsToWeights, 
    0 => ("a vector of weights", "1-D float tensor of weights by length")
}

args!{LengthsToWeights, 
    0 => ("power", "n of 1/pow(length,n) for normalization")
}

impl<Context> LengthsToWeightsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            power_(this->template GetSingleArgument<float>("power", 0.5))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto* input_data = input.template data<Index>();
        auto input_size = input.numel();
        auto* output = Output(0);

        int64_t output_size = 0;
        for (auto i = 0; i < input_size; i++) {
          CAFFE_ENFORCE_GE(input_data[i], 0, "unexpected negative length value");
          output_size += input_data[i];
        }

        std::function<float(const int64_t& length, const float& power)> getWeight;
        if (power_ == 0.5) {
          getWeight = [](const int64_t& length, const float& /*power*/) {
            return 1.0 / std::sqrt(length);
          };
        } else if (power_ == 1) {
          getWeight = [](const int64_t& length, const float& /*power*/) {
            return 1.0 / length;
          };
        } else {
          getWeight = [](const int64_t& length, const float& power) {
            return 1.0 / std::pow(length, power);
          };
        }

        output->Resize(output_size);
        auto* output_data = output->template mutable_data<float>();
        int64_t cnt = 0;
        for (auto i = 0; i < input_size; i++) {
          auto len = input_data[i];
          if (len == 0) {
            continue;
          }
          CAFFE_ENFORCE_LE(cnt + len, output_size, "unexpected lengths value");

          float weight_value = getWeight(len, power_);
          std::fill(output_data + cnt, output_data + cnt + len, weight_value);
          cnt += len;
        }

        return true;
        */
    }
}
