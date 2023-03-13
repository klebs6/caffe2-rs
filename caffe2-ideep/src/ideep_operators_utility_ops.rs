crate::ix!();


///----------------------------------
pub struct CopyCPUToIDEEPOp {

    //USE_IDEEP_DEF_ALIASES();
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

///-----------------------------------
pub struct IDEEPCopyOp {

    //USE_IDEEP_DEF_ALIASES();
    base: IDEEPOperator,
} 

impl IDEEPCopyOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = OperatorStorage::Input<itensor>(0);
        auto* Y = Output(0);
        if (X != *Y) {
          Y->reinit_like(X);
          ideep::direct_copy::compute(X, *Y);
        }

        return true;
        */
    }
}

///---------------------------------
pub struct CopyIDEEPToCPUOp {

    //USE_IDEEP_DEF_ALIASES();
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

///----------------------------------------

pub struct IDEEPWeightedSumOp {

    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

} 

impl IDEEPWeightedSumOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize() % 2, 0);
        auto ndims = Input(0).ndims();
        auto nelems = Input(0).get_nelems();
        auto w_nelems = Input(1).get_nelems();
        CAFFE_ENFORCE_GT(nelems, 0);
        CAFFE_ENFORCE_EQ(w_nelems, 1);
        auto* output = Output(0);
        std::vector<float> scales;
        scales.reserve(InputSize() / 2);
        std::vector<itensor> inputs;
        inputs.reserve(InputSize() / 2);
        for (int i = 0; i < InputSize(); i += 2) {
          auto& X = Input(i);
          CAFFE_ENFORCE(X.ndims() == ndims);
          CAFFE_ENFORCE(X.get_nelems() == nelems);
          CAFFE_ENFORCE(Input(i + 1).get_nelems() == w_nelems);
          inputs.push_back(X);
          auto scale = static_cast<float *>(Input(i + 1).get_data_handle());
          scales.push_back(scale[0]);
        }

        ideep::sum::compute(scales, inputs, *output);

        return true;
        */
    }
}

register_ideep_operator!{CopyCPUToIDEEP, CopyCPUToIDEEPOp}
register_ideep_operator!{CopyIDEEPToCPU, CopyIDEEPToCPUOp}
register_ideep_operator!{Copy, IDEEPCopyOp}
register_ideep_operator!{WeightedSum, IDEEPWeightedSumOp}
