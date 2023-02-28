use crate::{
    OperatorStorage
};

use crate::{
    OperatorDef,
    Workspace
};

crate::ix!();

/**
  Operator wrapper for generating int8 quant scheme blob given the 
  preserve sparsity and quantization kind
  */
pub struct Int8QuantSchemeBlobFillOp<Context,Engine> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<Engine>,
}

register_cpu_operator!{
    Int8QuantSchemeBlobFill,
    Int8QuantSchemeBlobFillOp<CPUContext, DefaultEngine>
}

num_inputs!{Int8QuantSchemeBlobFill, 0}

num_outputs!{Int8QuantSchemeBlobFill, 1}

outputs!{Int8QuantSchemeBlobFill, 
    0 => ("quant_scheme", "Int8QuantSchemeBlob that specifies the quantization kind and preserve_sparsity options when generating the quant params.")
}

args!{Int8QuantSchemeBlobFill, 
    0 => ("quantization_kind", "The kind of quant scheme that would be used to generate quant param"),
    1 => ("preserve_sparsity", "Flag to preserve sparsity or not")
}

tensor_inference_function!{Int8QuantSchemeBlobFill, /* ([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(TensorProto_DataType_STRING);
      out[0].add_dims(1);
      return out;
    }) */
}

impl<Context, Engine> Int8QuantSchemeBlobFillOp<Context,Engine> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            std::string quantization_kind =
            this->template GetSingleArgument<std::string>(
                "quantization_kind", "MIN_MAX_QUANTIZATION");
        bool preserve_sparsity =
            this->template GetSingleArgument<bool>("preserve_sparsity", false);

        auto* output_qscheme =
            this->template Output<unique_ptr<Int8QuantSchemeBlob>>(0);
        output_qscheme->reset(
            new Int8QuantSchemeBlob(quantization_kind, preserve_sparsity));
        return true;
        */
    }
}
