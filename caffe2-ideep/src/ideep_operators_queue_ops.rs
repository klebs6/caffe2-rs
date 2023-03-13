crate::ix!();


pub struct IDEEPCreateBlobsQueueOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    ws:   *mut Workspace,
    name: String,
}

impl IDEEPCreateBlobsQueueOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            ws_(ws),
            name(operator_def.output().Get(0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto capacity = GetSingleArgument("capacity", 1);
        const auto numBlobs = GetSingleArgument("num_blobs", 1);
        const auto enforceUniqueName =
            GetSingleArgument("enforce_unique_name", false);
        const auto fieldNames =
            OperatorStorage::template GetRepeatedArgument<std::string>("field_names");
        CAFFE_ENFORCE_EQ(this->OutputSize(), 1);
        auto queuePtr = OperatorStorage::Outputs()[0]
                            ->template GetMutable<std::shared_ptr<BlobsQueue>>();

        CAFFE_ENFORCE(queuePtr);
        *queuePtr = std::make_shared<BlobsQueue>(
            ws_, name, capacity, numBlobs, enforceUniqueName, fieldNames);
        return true;
        */
    }
}


///-------------------------------
pub struct IDEEPSafeEnqueueBlobsOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

}

impl IDEEPSafeEnqueueBlobsOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto queue =
            OperatorStorage::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queue);
        auto size = queue->getNumBlobs();
        CAFFE_ENFORCE(
            OutputSize() == size + 1,
            "Expected " + caffe2::to_string(size + 1) + ", " +
                " got: " + caffe2::to_string(size));
        bool status = queue->blockingWrite(OperatorStorage::Outputs());

        auto st = OperatorStorage::Output<TensorCPU>(1, CPU);
        st->Resize();
        auto stat = st->template mutable_data<bool>();
        stat[0] = !status;
        return true;
        */
    }
}

register_ideep_operator!{CreateBlobsQueue, IDEEPCreateBlobsQueueOp}
should_not_do_gradient!{IDEEPCreateBlobsQueueOp}

register_ideep_operator!{SafeEnqueueBlobs, IDEEPSafeEnqueueBlobsOp}
should_not_do_gradient!{IDEEPSafeEnqueueBlobsOp}
