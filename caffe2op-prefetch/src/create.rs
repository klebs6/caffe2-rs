crate::ix!();

impl<Context> PrefetchOperator<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : OperatorStorage(operator_def, ws),
            context_(operator_def.device_option()),
            prefetched_(false),
            prefetch_success_(true),
            finalize_(false),
            no_prefetch_(GetSingleArgument<bool>("no_prefetch", false)) 
        context_.SwitchToDevice();
        */
    }
}

impl<Context> Drop for PrefetchOperator<Context> {

    fn drop(&mut self) {
        todo!();
        /* 
        CHECK(finalize_ || !prefetch_thread_.get())
            << "YOU MADE A PROGRAMING ERROR: derived class of PrefetchOperator "
               "should call Finalize() in its destructor so the prefetching "
               "thread is joined. ";
       */
    }
}
