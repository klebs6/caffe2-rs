crate::ix!();

/**
  | Decode inputs using codebook.
  | 
  | This is a general LUT operator that returns
  | tensors with values from codebook (input
  | 0) based on given indices in codes (input
  | 1 ~ n).
  |
  */
#[USE_OPERATOR_FUNCTIONS("CPUContext")]
pub struct QuantDecodeGradientOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl QuantDecodeGradientOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Inputs: 1 codebook, n tensors of codes, and n corresponding gradients.
        CAFFE_ENFORCE(InputSize() >= 3 && InputSize() % 2 == 1);
        const int num_code_tensors = (InputSize() - 1) / 2;
        CAFFE_ENFORCE_EQ(OutputSize(), 1);

        const auto& codebook = Input(0);
        CAFFE_ENFORCE(codebook.template IsType<float>(), codebook.dtype().name());

        auto* gradient = Output(0, codebook.sizes(), at::dtype<float>());
        auto* gradient_ptr = gradient->template mutable_data<float>();
        std::fill(gradient_ptr, gradient_ptr + gradient->numel(), 0);

        for (int i = 0; i < num_code_tensors; i++) {
          auto& codes_i = Input(i + 1);
          auto& output_gradient_i = Input(i + num_code_tensors + 1);
          DecodeGeneral(codebook, codes_i, &output_gradient_i, gradient, false);
        }
        return true;
        */
    }
}
