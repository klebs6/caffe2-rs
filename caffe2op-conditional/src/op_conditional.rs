crate::ix!();

/**
  | Given a 1-D tensor of boolean values,
  | apply conditional operator along the
  | first dimension of DataT and DataF and 
  | return DataO. Note, DataT and
  | DataF must have the exact same shape
  | and type.
  |
  */
pub struct ConditionalOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

}

register_cpu_operator!{Conditional, ConditionalOp<CPUContext>}

num_inputs!{Conditional, 3}

num_outputs!{Conditional, 1}

inputs!{Conditional, 
    0 => ("Condition", "Boolean tensor to select DataT or DataF"),
    1 => ("DataT", "Data to use when True"),
    2 => ("DataF", "Data to use when False")
}

outputs!{Conditional, 
    0 => ("DataO", "Output data after applying ConditionalOp")
}

identical_type_and_shape_of_input!{Conditional, 1}

no_gradient!{Conditional}

impl<Context> ConditionalOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

impl ConditionalOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& condition = Input(0);
      auto& dataT = Input(1);
      auto& dataF = Input(2);

      // verify the inputs shape
      CAFFE_ENFORCE_EQ(condition.dim(), 1);
      CAFFE_ENFORCE(dataT.dim() >= 1);
      CAFFE_ENFORCE(dataT.sizes()[0] == condition.sizes()[0]);
      CAFFE_ENFORCE_EQ(dataT.dim(), dataF.dim());
      for (size_t i = 0; i < dataT.sizes().size(); i++) {
        CAFFE_ENFORCE(dataT.sizes().at(i) == dataF.sizes().at(i));
      }
      const auto innerSize = dataT.size_from_dim(1);
      const auto innerSizeBytes = innerSize * dataT.dtype().itemsize();
      CAFFE_ENFORCE(innerSize * dataF.dtype().itemsize() == innerSizeBytes);

      // initialize output shape
      auto* dataOut = Output(0);
      const auto* condPtr = condition.template data<bool>();
      dataOut->ResizeLike(dataT);
      auto* outPtr = (char*)dataOut->raw_mutable_data(dataT.dtype());

      // perform conditional op along first dimension
      const auto* ptrT = (char*)dataT.raw_data();
      const auto* ptrF = (char*)dataF.raw_data();
      for (int64_t i = 0; i < condition.numel(); i++) {
        auto* dst = outPtr + i * innerSizeBytes;
        if (condPtr[i]) {
          context_.CopyItemsSameDevice(
              dataT.dtype(), innerSize, ptrT + i * innerSizeBytes, dst);
        } else {
          context_.CopyItemsSameDevice(
              dataF.dtype(), innerSize, ptrF + i * innerSizeBytes, dst);
        }
      }
      return true;
        */
    }
}
