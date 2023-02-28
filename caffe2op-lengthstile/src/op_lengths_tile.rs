crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    Tensor,
    OperatorDef,
    CPUContext
};

#[test] fn lengths_tile_op_example() {

    todo!();
    /*
    Example:
      DATA  = [
          [1.0, 1.2],
          [2.3, 3.4],
          [4.5, 5.7],
          [6.8, 7.9],
      ]
      LENGTHS = [0, 1, 3, 2]
      OUTPUT = [
          [2.3, 3.4],
          [4.5, 5.7],
          [4.5, 5.7],
          [4.5, 5.7],
          [6.8, 7.9],
          [6.8, 7.9],
      ]
    */
}

/**
  | Given DATA tensor of rank r >= 1, and LENGTHS
  | tensor of rank 1, duplicate each entry
  | of the outer-most dimension of DATA
  | according to LENGTHS, and concatenate
  | them in an output tensor of rank r.
  |
  */
pub struct LengthsTileOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    lengths_host:       Tensor, // default = CPU
    row_mapping_host:   Tensor,
    row_mapping_device: Tensor,
}

register_cpu_operator!{LengthsTile, LengthsTileOp<CPUContext>}

num_inputs!{LengthsTile, 2}

num_outputs!{LengthsTile, 1}

inputs!{LengthsTile, 
    0 => ("DATA",    "Tensor of rank r >= 1. First dimension must be equal to the size of lengths"),
    1 => ("LENGTHS", "Tensor of int32 lengths of rank 1")
}

outputs!{LengthsTile, 
    0 => ("OUTPUT", "Tensor of rank r")
}

input_tags!{
    LengthsTileOp {
        Data,
        Lengths
    }
}

impl<Context> LengthsTileOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

impl LengthsTileOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(DATA);
      auto& lengths = Input(LENGTHS);
      auto* output = Output(0);

      CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be 1-D");
      CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
      CAFFE_ENFORCE_EQ(lengths.numel(), data.size(0));

      // Context::CopyFrom and math::Sum need the same context to avoid race
      // conditions
      // why? CPUContext is not used in Sum
      lengths_host_.CopyFrom(lengths); // sync copy
      auto lengths_size = lengths_host_.numel();
      auto* lengths_data = lengths_host_.data<int32_t>();

      int32_t total_length = 0;
      CPUContext cpuContext;
      math::Sum<int32_t, CPUContext>(
          lengths_size, lengths_data, &total_length, &cpuContext);

      auto shape = data.sizes().vec();
      shape[0] = total_length;
      output->Resize(shape);

      auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
      auto src = static_cast<const char*>(data.raw_data());
      auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

      for (int64_t i = 0; i < lengths_size; ++i) {
        auto length = lengths_data[i];
        CAFFE_ENFORCE_GE(length, 0);
        for (int32_t j = 0; j < length; ++j) {
          context_.CopyBytesSameDevice(block_bytesize, src, out);
          out += block_bytesize;
        }
        src += block_bytesize;
      }
      return true;
        */
    }
}

pub struct GetLengthsTileGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLengthsTileGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);
        return SingleGradientDef(
            "LengthsSum",
            "",
            // input 1 is the lengths used to repeat
            // DATA in the forward pass
            vector<string>{GO(0), I(1)},
            // only concerned with the gradient on "DATA"
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LengthsTile, GetLengthsTileGradient}
