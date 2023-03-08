crate::ix!();

#[USE_DISPATCH_HELPER]
pub struct PartitionOp {
    base: PartitionOpBase,
}

impl PartitionOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : PartitionOpBase(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            ApplyPartition<Index>(false /* skipFirstArgument */);
        return true;
        */
    }
}
