crate::ix!();

pub struct BaseReducerMeta {
    block_size:   i64,
    block_shape:  Vec<i64>,
    first_dim:    bool,
}

impl BaseReducerMeta {
    
    pub fn new(first: Option<bool>) -> Self {
    
        let first: bool = first.unwrap_or(true);

        todo!();
        /*
            : first_dim(first)
        */
    }
    
    #[inline] pub fn compute_meta(
        &mut self, 
        dims: &[i32], 
        skip_dims: usize)  
    {
        todo!();
        /*
            first_dim ? block_shape.assign(dims.begin() + skip_dims, dims.end())
                : block_shape.assign(dims.begin(), dims.end() - skip_dims);
            block_size = first_dim ? size_from_dim_(skip_dims, dims)
                : size_from_dim_(dims.size() - skip_dims, dims);
        */
    }
    
    #[inline] pub fn observe_input(
        &mut self, 
        input:     i32,
        value:     &Tensor,
        skip_dims: i32)  
    {
        todo!();
        /*
            DCHECK_EQ(0, input);
            auto dims = value.sizes();
            computeMeta(dims, skip_dims);
        */
    }
    
    #[inline] pub fn append_output_shape(
        &mut self, 
        output_shape: *mut Vec<i64>)  
    {
        todo!();
        /*
            output_shape->insert(
                output_shape->end(), block_shape.begin(), block_shape.end());
        */
    }
    
    #[inline] pub fn get_output_shape(
        &mut self, 
        input: &TensorShape, 
        skip_dims: i32) -> Vec<i64> 
    {
        todo!();
        /*
            vector<int64_t> dims(in.dims().begin(), in.dims().end());
            computeMeta(dims, skip_dims);
            return block_shape;
        */
    }
}
