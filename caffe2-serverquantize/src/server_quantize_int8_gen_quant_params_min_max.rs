crate::ix!();

use crate::{
    Operator,
    Workspace,
    OperatorDef,
    OperatorStorage,
};

/**
  | Operator wrapper for generating int8
  | tensor quantization parameters given
  | lower and upper bound of the input tensor
  |
  */
pub struct Int8GenQuantParamsMinMaxOp<Context,Engine> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<Engine>,
}

register_cpu_operator!{
    Int8GenQuantParamsMinMax,
    Int8GenQuantParamsMinMaxOp<CPUContext, DefaultEngine>
}

num_inputs!{Int8GenQuantParamsMinMax, (2,3)}

num_outputs!{Int8GenQuantParamsMinMax, 1}

inputs!{Int8GenQuantParamsMinMax, 
    0 => ("min",           "The lower bound of the tensor to be quantized."),
    1 => ("max",           "The upper bound of the tensor to be quantized."),
    2 => ("quant_scheme",  "(Optional) Int8QuantSchemeBlob that specifies the quantization kind and preserve_sparsity options when generating the quant params. We only use preserve_sparsity in this op which is default to be false.")
}

outputs!{Int8GenQuantParamsMinMax, 
    0 => ("quant_param",   "Int8QuantParamsBlob that contains the scale and zero_point info in TensorQuantizationParams type.")
}

tensor_inference_function!{Int8GenQuantParamsMinMax, /* ([](const OperatorDef& /* def */,
                                const vector<TensorShape>& /* in */) {
      vector<TensorShape> out(1);
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      out[0].add_dims(1);
      return out;
    }) */
}

impl<Context,Engine> Int8GenQuantParamsMinMaxOp<Context,Engine> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Generate Int8 quant params based on the input data (last N samples of the
        // activations) and the quant scheme
        const float min =
              OperatorStorage::Input<Tensor>(0, CPU).template data<float>()[0];
        const float max =
              OperatorStorage::Input<Tensor>(1, CPU).template data<float>()[0];
        bool preserve_sparsity = false;
        if (InputSize() == 3){
            const auto* quant_scheme =
            this->template Input<unique_ptr<Int8QuantSchemeBlob>>(2).get();
            preserve_sparsity = quant_scheme->preserve_sparsity_;
        }
        dnnlowp::QuantizationFactory* qfactory =
            dnnlowp::QuantizationFactory::GetDefaultInstance();
        TensorQuantizationParams qparam = qfactory->ChooseQuantizationParams(
            min,
            max,
            8,
            preserve_sparsity);
        auto* output_qparam =
            this->template Output<unique_ptr<Int8QuantParamsBlob>>(0);
        output_qparam->reset(
            new Int8QuantParamsBlob(qparam.scale, qparam.zero_point));
        LOG_EVERY_N(INFO, 1) << "scale and bias are " << qparam.scale << "," << qparam.zero_point;
        return true;
        */
    }
}
