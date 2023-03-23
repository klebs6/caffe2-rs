crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPCreateBlobsQueueOp {
    base: IDEEPOperator,
    ws:   *mut Workspace,
    name: String,
}

should_not_do_gradient!{IDEEPCreateBlobsQueueOp}

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
