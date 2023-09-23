crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalNeuronType.h]

pub enum NeuronType {
    None,
    Clamp,
    Relu,
    Sigmoid,
    HardSigmoid,
    Tanh,
}

#[inline] pub fn neuron_type_with_range(
        output_min: Option<Scalar>,
        output_max: Option<Scalar>) -> NeuronType {
    
    todo!();
        /*
            float inf_max = numeric_limits<float>::infinity();
      float inf_min = -numeric_limits<float>::infinity();
      float output_max_ =
          output_max.has_value() ? output_max.value().toFloat() : inf_max;
      float output_min_ =
          output_min.has_value() ? output_min.value().toFloat() : inf_min;
      if (output_max_ == inf_max && output_min_ == 0) {
        return NeuronType::Relu;
      } else if (output_max_ < inf_max && output_min_ > inf_min) {
        return NeuronType::Clamp;
      } else {
        return NeuronType::None;
      }
        */
}

#[inline] pub fn neuron_type(ty: NeuronType) -> *mut MPSCNNNeuron {
    
    todo!();
        /*
            if (type == NeuronType::Relu) {
        return [MPSCNNNeuronOp relu];
      } else if (type == NeuronType::Sigmoid) {
        return [MPSCNNNeuronOp sigmoid];
      } else if (type == NeuronType::Tanh) {
        return [MPSCNNNeuronOp tanh];
      } else if (type == NeuronType::HardSigmoid) {
        if (@available(iOS 11.0, *)) {
          return [MPSCNNNeuronOp hardSigmoid];
        } else {
          return nil;
        }
      } else {
        return nil;
      }
        */
}
