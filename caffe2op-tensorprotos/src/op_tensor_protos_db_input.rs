crate::ix!();

/**
  | TensorProtosDBInput is a simple input
  | operator that basically reads things
  | from a db where each key-value pair stores
  | an index as key, and a TensorProtos object
  | as value. These TensorProtos objects
  | should have the same size, and they will
  | be grouped into batches of the given
  | size. The DB Reader is provided as input to 
  | the operator and it returns as many output 
  | tensors as the size of the
  | 
  | TensorProtos object. Each output will
  | simply be a tensor containing a batch
  | of data with size specified by the 'batch_size'
  | argument containing data from the corresponding
  | index in the
  | 
  | TensorProtos objects in the DB.
  |
  */
pub struct TensorProtosDBInput<Context> {

    base:              PrefetchOperator<Context>,

    /**
      | Prefetch will always just happen on
      | the CPU side.
      |
      */
    prefetched_blobs:  Vec<Blob>,
    batch_size:        i32,
    shape_inferred:    bool, // default = false
    key:               String,
    value:             String,
}

register_cpu_operator!{TensorProtosDBInput, TensorProtosDBInput<CPUContext>}

num_inputs!{TensorProtosDBInput, 1}

num_outputs!{TensorProtosDBInput, (1,INT_MAX)}

inputs!{TensorProtosDBInput, 
    0 => ("data", "A pre-initialized DB reader. Typically, this is obtained by calling CreateDB operator with a db_name and a db_type. The resulting output blob is a DB Reader tensor")
}

outputs!{TensorProtosDBInput, 
    0 => ("output", "The output tensor in which the batches of data are returned. The number of output tensors is equal to the size of (number of TensorProto's in) the TensorProtos objects stored in the DB as values. Each output tensor will be of size specified by the 'batch_size' argument of the operator")
}

args!{TensorProtosDBInput, 
    0 => ("batch_size", "(int, default 0) the number of samples in a batch. The default value of 0 means that the operator will attempt to insert the entire data in a single output blob.")
}

no_gradient!{TensorProtosDBInput}

register_cuda_operator!{TensorProtosDBInput, TensorProtosDBInput<CUDAContext>}

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
    
    #[inline] pub fn prefetch(&mut self) -> bool {
        
        todo!();
        /*
            const db::DBReader& reader = this->template Input<db::DBReader>(0);
      TensorDeserializer deserializer;
      if (batch_size_ == 0) {
        // We do not need to construct a batch. As a result, we will simply
        // deserialize everything into the target prefetched blob.
        reader.Read(&key_, &value_);
        TensorProtos protos;
        CAFFE_ENFORCE(protos.ParseFromString(value_));
        CAFFE_ENFORCE(protos.protos_size() == OutputSize());
        for (int i = 0; i < protos.protos_size(); ++i) {
          if (protos.protos(i).has_device_detail()) {
            protos.mutable_protos(i)->clear_device_detail();
          }
          BlobSetTensor(
              &prefetched_blobs_[i], deserializer.Deserialize(protos.protos(i)));
          // deserializer.Deserialize(
          //     protos.protos(i), BlobGetMutableTensor(&prefetched_blobs_[i],
          //     CPU));
        }
      } else {
        for (int item_id = 0; item_id < batch_size_; ++item_id) {
          reader.Read(&key_, &value_);
          TensorProtos protos;
          CAFFE_ENFORCE(protos.ParseFromString(value_));
          CAFFE_ENFORCE(protos.protos_size() == OutputSize());
          // Note: shape_inferred_ is ignored, we'll always get dimensions from
          // proto
          for (int i = 0; i < protos.protos_size(); ++i) {
            vector<int64_t> dims(
                protos.protos(i).dims().begin(), protos.protos(i).dims().end());
            dims.insert(dims.begin(), batch_size_);
            if (protos.protos(i).has_device_detail()) {
              protos.mutable_protos(i)->clear_device_detail();
            }
            Tensor src = deserializer.Deserialize(protos.protos(i));
            Tensor* dst = BlobGetMutableTensor(
                &prefetched_blobs_[i], dims, at::dtype(src.dtype()).device(CPU));
            DCHECK_EQ(src.numel() * batch_size_, dst->numel());
            this->context_.CopyItemsSameDevice(
                src.dtype(),
                src.numel(),
                src.raw_data(),
                static_cast<char*>(dst->raw_mutable_data(src.dtype())) +
                    src.nbytes() * item_id);
          }
        }
      }
      return true;
        */
    }
}

impl<Context> Drop for TensorProtosDBInput<Context> {
    fn drop(&mut self) {
        todo!();
        /* 
        PrefetchOperator<Context>::Finalize();
       */
    }
}
