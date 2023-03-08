crate::ix!();

pub struct BaseReducerGradientMeta {
    block_size:   i64,
    block_shape:  Vec<i64>,
    first_dim:    bool,
}

impl BaseReducerGradientMeta {
    
    pub fn new(
        out_grad:  &Tensor,
        skip_dims: i32,
        first_dim: Option<bool>) -> Self {

        let first_dim: bool = first_dim.unwrap_or(true);

        todo!();
        /*
            : first_dim(first_dim) 

                auto dims = out_grad.sizes();
                first_dim ? block_shape.assign(dims.begin() + skip_dims, dims.end())
                    : block_shape.assign(dims.begin(), dims.end() - skip_dims);
                block_size = first_dim
                    ? out_grad.size_from_dim(skip_dims)
                    : out_grad.size_from_dim(out_grad.dim() - skip_dims);
        */
    }

    /// optional grad to populate
    #[inline] pub fn observe_original_input(
        &mut self, 
        original_input: i32,
        value:          &Tensor,
        input_grad:     *mut Tensor,
        skip_dims:      i32)  {

        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn append_grad_shape(&mut self, output_shape: *mut Vec<i64>)  {
        
        todo!();
        /*
            output_shape->insert(
                output_shape->end(), block_shape.begin(), block_shape.end());
        */
    }
}
