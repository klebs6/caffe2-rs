crate::ix!();

///-----------------------------------------------
///Get the size of the input vector
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TensorVectorSizeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{TensorVectorSize, 1}

num_outputs!{TensorVectorSize, 1}

inputs!{TensorVectorSize, 
    0 => ("tensor vector", "std::unique_ptr<std::vector<Tensor> >")
}

outputs!{TensorVectorSize, 
    0 => ("size", "int32_t size")
}

impl<Context> TensorVectorSizeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& vector_ptr = OperatorStorage::Input<TensorVectorPtr>(TENSOR_VECTOR);
        auto* size = Output(SIZE);
        size->Resize();
        // 32-bit should be enough here
        *size->template mutable_data<int32_t>() = vector_ptr->size();
        return true;
        */
    }
}

input_tags!{
    TensorVectorSizeOp {
        TensorVector
    }
}

output_tags!{
    TensorVectorSizeOp {
        Size
    }
}

