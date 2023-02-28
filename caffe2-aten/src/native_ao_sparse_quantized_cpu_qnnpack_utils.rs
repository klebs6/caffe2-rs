crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h]

/**
  | TODO: Refacto qnnpack_utils.h so as
  | to separate code needed for quantized
  | op from the generic qnnpack specific
  | quantization utilities.
  |
  */
#[cfg(USE_PYTORCH_QNNPACK)]
pub struct PackedLinearWeightQnnp {

    base:                  LinearPackedParamsBase,

    orig_weight:           Tensor,
    orig_bias:             Option<Tensor>,

    /**
      | Seperate copy of bias exist so that we
      | can fill in zeros when optional bias
      | does not exist. This is to compy with
      | qnnpack operator that expects bias
      | to be present.
      | 
      | In case bias is present bias_ is just
      | a reference to orig_bias_
      |
      */
    bias:                  Tensor,

    q_scheme:              QScheme,
    input_scale:           f64,
    bcsr_matrix:           Box<QnnpackBCSRMatrix>,
    w_scales:              Tensor,
    w_zero_points:         Vec<u8>,
    requantization_scales: Vec<f32>,
    sparse_linear_op:      Box<PytorchQnnpOperator,QnnpackOperatorDeleter>, // default = { nullptr }
}

impl PackedLinearWeightQnnp {

    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn new(
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,

        /* block sparsity size across output_features */
        out_features_block_size: i64,

        /* block sparsity size across input_features */
        in_features_block_size:  i64) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
            false, "Static quantized sparse linear unimplemented on QNNPACK");
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
            false, "Static quantized sparse linear unimplemented on QNNPACK");
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic_relu(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn unpack(&mut self) -> LinearPackedSerializationType {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn bias(&mut self) -> Option<Tensor> {
        
        todo!();
        /*
            return orig_bias_;
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn prepack(
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,
        out_features_block_size: i64,
        in_features_block_size:  i64) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: &Tensor) -> Tensor {
    
        todo!();
        /*
        
        */
    }
}
