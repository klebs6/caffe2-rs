crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPSafeEnqueueBlobsOp {
    base: IDEEPOperator,
}

should_not_do_gradient!{IDEEPSafeEnqueueBlobsOp}

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
