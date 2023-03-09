crate::ix!();

pub struct BaseInputAccessor<TData> {
    data:  *const c_void, // default = nullptr
    phantom: PhantomData<TData>,
}

impl<TData> BaseInputAccessor<TData> {
    
    #[inline] pub fn observe_input(&mut self, data_input: &Tensor) -> bool {
        
        todo!();
        /*
            data_ = dataInput.raw_data();
        return dataInput.template IsType<TData>();
        */
    }
    
    #[inline] pub fn get_block_ptr(&mut self, 
        in_block_size: i64,
        idx:           i64,
        blocks:        Option<i64>) -> *const TData {
        let blocks: i64 = blocks.unwrap_or(1);

        todo!();
        /*
            return static_cast<const TData*>(data_) + in_block_size * idx;
        */
    }
}
