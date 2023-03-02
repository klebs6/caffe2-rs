crate::ix!();

///------------------------------------------
///Create a std::unique_ptr<std::vector<Tensor> >
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CreateTensorVectorOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{CreateTensorVector, 0}

num_outputs!{CreateTensorVector, 1}

impl<Context> CreateTensorVectorOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto ptr = make_unique<std::vector<Tensor>>();
        *OperatorStorage::Output<TensorVectorPtr>(TENSOR_VECTOR) = std::move(ptr);
        return true;
        */
    }
}

output_tags!{
    CreateTensorVectorOp {
        TensorVector
    }
}

