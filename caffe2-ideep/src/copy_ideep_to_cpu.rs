crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
pub struct CopyIDEEPToCPUOp {
    base: IDEEPOperator,
} 

num_inputs!{CopyIDEEPToCPU, 1}

num_outputs!{CopyIDEEPToCPU, 1}

inputs!{CopyIDEEPToCPU, 
    0 => ("ideep_blob", "The input IDEEP tensort to copy")
}

outputs!{CopyIDEEPToCPU, 
    0 => ("cpu_blob", "The output TensorCPU to copy to")
}

impl CopyIDEEPToCPUOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input_blob = OperatorStorage::InputBlob(0);
        if (BlobIsTensorType(input_blob, CPU)) {
          VLOG(2) << "Directing sharing of TensorCPU";
          const auto& X = OperatorStorage::Input<Tensor>(0, CPU);
          OutputTensorCopyFrom(0, at::device(CPU), X);
        } else {
          const auto& X = OperatorStorage::Input<itensor>(0);
          if (X.get_data_type() == itensor::data_type::f32) {
            std::vector<int64_t> dims;
            for (int i = 0; i < X.get_dims().size(); ++i) {
              dims.push_back(X.get_dims()[i]);
            }
            auto* Y =
                OperatorStorage::OutputTensor(0, dims, at::dtype<float>().device(CPU));
            X.to_public(Y->template mutable_data<float>());
          } else {
            CAFFE_THROW("Unsupported ideep type: ",
                        static_cast<int>(X.get_data_type()));
          }
        }
        return true;
        */
    }
}
