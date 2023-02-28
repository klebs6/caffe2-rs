crate::ix!();

use crate::{
    TensorQuantizationParams,
    OperatorStorage,
    Blob,
    Workspace,
    QuantizationFactory
};

/**
  | Wrap a floating-point operator with
  | quantized inputs with type T.
  | 
  | This class is to measure quantization
  | error against fp32 reference.
  |
  */
pub struct OpWrapper<OpType,T> {

    /// container quantized op
    op:                  *mut OperatorStorage,

    local_ws:            Workspace,
    local_input_blobs:   Vec<*mut Blob>,
    local_output_blobs:  Vec<*mut Blob>,

    /// contained fp32 reference op
    local_op:            Box<OpType>,
    qfactory:            *mut QuantizationFactory,
    phantom: PhantomData<T>,
}

impl<OpType,T> OpWrapper<OpType,T> {
    
    pub fn new(op: *mut OperatorStorage, qfactory: *mut QuantizationFactory) -> Self {
    
        todo!();
        /*
            : op_(op), qfactory_(qfactory) 

        for (auto name : op->debug_def().input()) {
          local_input_blobs_.push_back(local_ws_.CreateBlob(name));
          CHECK_NOTNULL(local_input_blobs_.back());
        }
        OperatorDef def = op->debug_def();
        local_op_.reset(new OpType(def, &local_ws_));
        for (auto name : def.output()) {
          local_output_blobs_.push_back(local_ws_.GetBlob(name));
          CHECK_NOTNULL(local_output_blobs_.back());
        }
        */
    }
    
    #[inline] pub fn dequantize_input(&mut self)  {
        
        todo!();
        /*
            const OperatorDef& def = op_->debug_def();
        CPUContext context(def.device_option());

        for (int i = 0; i < op_->InputSize(); ++i) {
          if (op_->InputIsType<int8::Int8TensorCPU>(i)) {
            const TensorCPU& qtensor = op_->Input<int8::Int8TensorCPU>(i).t;
            TensorCPU* float_tensor =
                BlobGetMutableTensor(local_input_blobs_[i], CPU);
            // FIXME: doesn't work for bias so we shouldn't quantize bias before
            // model loading when we're running a shadow operator in fp32 for
            // example for measuring quantization error.
            float_tensor->ResizeLike(qtensor);
            fbgemm::Dequantize<T>(
                qtensor.data<T>(),
                float_tensor->template mutable_data<float>(),
                qtensor.numel(),
                dnnlowp::GetInputTensorQuantizationParamsOf(op_, i, qfactory_));
          } else {
            local_input_blobs_[i]->ShareExternal(
                const_cast<void*>(op_->Inputs()[i]->GetRaw()),
                op_->Inputs()[i]->meta());
          }
        }
        */
    }
    
    #[inline] pub fn get(&mut self) -> *mut OpType {
        
        todo!();
        /*
            return local_op_.get();
        */
    }
    
    #[inline] pub fn get_output_quantization_params(
        &mut self, 
        qfactory: *mut QuantizationFactory, 
        index: Option<i32>) -> TensorQuantizationParams 
    {
        let index: i32 = index.unwrap_or(0);

        todo!();
        /*
            using namespace dnnlowp;

        float min, max;
        auto& out_tensor = local_output_blobs_[index]->template Get<TensorCPU>();
        fbgemm::FindMinMax(
            out_tensor.template data<float>(), &min, &max, out_tensor.numel());
        if (op_->OperatorStorage::GetSingleArgument<std::string>("followed_by", "") ==
            "Relu") {
          min = std::max(0.0f, min);
          max = std::max(0.0f, max);
        }

        return qfactory->ChooseQuantizationParams(min, max);
        */
    }
}
