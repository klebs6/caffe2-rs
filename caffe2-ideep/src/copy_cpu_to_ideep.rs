crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
pub struct CopyCPUToIDEEPOp {
    base: IDEEPOperator,
} 

num_inputs!{CopyCPUToIDEEP, 1}

num_outputs!{CopyCPUToIDEEP, 1}

inputs!{CopyCPUToIDEEP, 
    0 => ("cpu_blob", "The input TensorCPU to copy")
}

outputs!{CopyCPUToIDEEP, 
    0 => ("ideep_blob", "The output IDEEP tensort to copy to")
}

impl CopyCPUToIDEEPOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = OperatorStorage::Input<Tensor>(0, CPU);
        auto* Y = OperatorStorage::OutputBlob(0);
        itensor::dims src_dims(X.sizes().begin(), X.sizes().end());
        if (!(Y->template IsType<itensor>() &&
              Y->Get<itensor>().get_data_type() == itensor::data_type::f32) ||
            Y->Get<itensor>().get_dims() != src_dims) {
          Y->Reset(new itensor());
          Y->GetMutable<itensor>()->resize(src_dims, itensor::data_type::f32);
        }
        Y->GetMutable<itensor>()->feed_from(
            src_dims, itensor::data_type::f32, X.raw_data());
        return true;
        */
    }
}
