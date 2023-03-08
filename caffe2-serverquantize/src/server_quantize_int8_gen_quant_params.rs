crate::ix!();

use crate::{
    TensorQuantizationParams,
    Workspace,
    OperatorStorage,
    OperatorDef
};

pub struct Int8QuantSchemeBlob {
    quantization_kind:  String,
    preserve_sparsity:  bool,
}

impl Int8QuantSchemeBlob {

    pub fn new(quantization_kind: String, preserve_sparsity: bool) -> Self {
    
        todo!();
        /*
            : quantization_kind_(quantization_kind),
            preserve_sparsity_(preserve_sparsity)
        */
    }
}

///---------------
pub struct Int8QuantParamsBlob {
    qparam:  TensorQuantizationParams,
}

impl Int8QuantParamsBlob {
    
    pub fn new(scale: f32, zero_point: i32) -> Self {
    
        todo!();
        /*
            qparam.scale = scale;
        qparam.zero_point = zero_point;
        */
    }
}

/**
  | Operator wrapper for generating int8
  | tensor quantization parameters given
  | the input data and quant scheme
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct Int8GenQuantParamsOp<Context,Engine> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<Engine>,
}

register_cpu_operator!{
    Int8GenQuantParams,
    Int8GenQuantParamsOp<CPUContext, DefaultEngine>
}

num_inputs!{Int8GenQuantParams, 2}

num_outputs!{Int8GenQuantParams, 1}

inputs!{Int8GenQuantParams, 
    0 => ("X", "The input data, or last N samples of the output activations."),
    1 => ("quant_scheme", "Int8QuantSchemeBlob that specifies the quantization kind and preserve_sparsity options when generating the quant params.")
}

outputs!{Int8GenQuantParams, 
    0 => ("quant_param", "Int8QuantParamsBlob that contains the scale and zero_point info in TensorQuantizationParams type.")
}

tensor_inference_function!{Int8GenQuantParams, /* ([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      out[0].add_dims(1);
      return out;
    }) */
}

impl<Context, Engine> Int8GenQuantParamsOp<Context,Engine> {

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
        const auto& input_data = Input(0);
        const auto* quant_scheme =
            this->template Input<unique_ptr<Int8QuantSchemeBlob>>(1).get();
        CAFFE_ENFORCE(input_data.dim() > 0);
        CAFFE_ENFORCE(quant_scheme);
        std::string quant_kind = quant_scheme->quantization_kind_;
        bool preserve_sparsity = quant_scheme->preserve_sparsity_;
        dnnlowp::QuantizationFactory* qfactory =
            dnnlowp::QuantizationFactory::GetDefaultInstance();
        TensorQuantizationParams qparam = qfactory->ChooseQuantizationParams(
            input_data.template data<float>(),
            input_data.numel(),
            dnnlowp::StringToKind(quant_kind),
            8,
            preserve_sparsity);
        auto* output_qparam =
            this->template Output<unique_ptr<Int8QuantParamsBlob>>(0);
        output_qparam->reset(
            new Int8QuantParamsBlob(qparam.scale, qparam.zero_point));
        return true;
        */
    }
}

// Expilictly register TypeMeta
caffe_known_type!{unique_ptr<Int8QuantSchemeBlob>}

caffe_known_type!{unique_ptr<Int8QuantParamsBlob>}
