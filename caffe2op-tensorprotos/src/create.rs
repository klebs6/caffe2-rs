crate::ix!();

impl<Context> Drop for TensorProtosDBInput<Context> {

    fn drop(&mut self) {
        todo!();
        /* 
        PrefetchOperator<Context>::Finalize();
       */
    }
}

impl<Context> TensorProtosDBInput<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : PrefetchOperator<Context>(operator_def, ws),
          prefetched_blobs_(operator_def.output_size()),
          batch_size_( this->template GetSingleArgument<int>("batch_size", 0))
        */
    }
    
    #[inline] pub fn copy_prefetched(&mut self) -> bool {
        
        todo!();
        /*
            for (int i = 0; i < OutputSize(); ++i) {
        OperatorStorage::template Output<Tensor>(i, Context::GetDeviceType())
            ->CopyFrom(
                prefetched_blobs_[i].template Get<TensorCPU>(), /* async */ true);
      }
      return true;
        */
    }
}
