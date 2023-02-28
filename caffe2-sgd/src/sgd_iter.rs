crate::ix!();

use crate::{
    Blob,
    BlobProto,
    BlobSerializationOptions,
    BlobSerializerBase,
    Operator,
    OperatorDef,
    SerializationAcceptor,
    TensorCPU,
    TypeMeta,
    Workspace,
};

#[inline] pub fn increment_iter(output: *mut TensorCPU)  {
    
    todo!();
    /*
        CAFFE_ENFORCE_EQ(
          output->numel(),
          1,
          "The output of IterOp exists, but not of the right size.");
      int64_t* iter = output->template mutable_data<int64_t>();
      CAFFE_ENFORCE(*iter >= 0, "Previous iteration number is negative.");
      CAFFE_ENFORCE(
          *iter < int64_t::max, "Overflow will happen!");
      (*iter)++;
    */
}

/**
  | Stores a singe integer, that gets incremented
  | on each call to Run().
  | 
  | Useful for tracking the iteration count
  | during SGD, for example.
  | 
  | IterOp runs an iteration counter. I
  | cannot think of a case where we would
  | need to access the iter variable on device,
  | so this will always produce a tensor
  | on the CPU side. If the blob already exists
  | and is a tensor<int64_t> object, we
  | will simply increment it (this emulates
  | the case when we want to resume training).
  | Otherwise we will have the iter starting
  | with 0.
  |
  */
pub struct IterOp<Context> {
    context: Context,

}

impl<Context> Operator for IterOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

}

num_inputs!{Iter, (0,1)}

num_outputs!{Iter, 1}

enforce_inplace!{Iter, vec![(0, 0)]}

no_gradient!{Iter}

impl<Context> IterOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() == 0) {
          VLOG(1) << "[Input size is zero]";
          if (!OperatorStorage::OutputIsTensorType(0, CPU)) {
            // This is the first run; set the iter to start with 0.
            LOG(ERROR) << "You are using an old definition of IterOp that will "
                          "be deprecated soon. More specifically, IterOp now "
                          "requires an explicit in-place input and output.";

            VLOG(1) << "Initializing iter counter.";
            auto* output = OperatorStorage::OutputTensor(
                0, {1}, at::dtype<int64_t>().device(CPU));
            output->template mutable_data<int64_t>()[0] = 0;
          }
        }
        IncrementIter(OperatorStorage::Output<Tensor>(0, CPU));
        return true;
        */
    }
}

/**
  | Similar to Iter, but takes a mutex as
  | the first input to make sure that updates
  | are carried out atomically. This can
  | be used in e.g. Hogwild sgd algorithms.
  |
  */
pub struct AtomicIterOpStats {
    /*
    CAFFE_STAT_CTOR(AtomicIterOpStats);
    CAFFE_EXPORTED_STAT(num_iter);
    */
}

pub struct AtomicIterOp<Context> {
    context: Context,
    stats: AtomicIterOpStats,
}

impl<Context> Operator for AtomicIterOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
}

num_inputs!{AtomicIter, 2}

num_outputs!{AtomicIter, 1}

inputs!{AtomicIter, 
    0 => ("mutex", "The mutex used to do atomic increment."),
    1 => ("iter", "The iter counter as an int64_t TensorCPU.")
}

identical_type_and_shape_of_input!{AtomicIter, 1}

enforce_inplace!{AtomicIter, vec![(1, 0)]}

no_gradient!{AtomicIter}

impl<Context> AtomicIterOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            stats_(std::string("atomic_iter/stats/") + operator_def.input(1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mutex = OperatorStorage::Input<std::unique_ptr<std::mutex>>(0);
        std::lock_guard<std::mutex> lg(*mutex);
        IncrementIter(OperatorStorage::Output<Tensor>(0, CPU));
        CAFFE_EVENT(stats_, num_iter);
        return true;
        */
    }
}

///-------------------------------
pub struct MutexSerializer { }

impl BlobSerializerBase for MutexSerializer {
    
    /**
      | Serializes a std::unique_ptr<std::mutex>.
      | Note that this blob has to contain std::unique_ptr<std::mutex>,
      | otherwise this function produces a
      | fatal error.
      |
      */
    #[inline] fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  {

        todo!();
        /*
          CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<std::mutex>>());
          BlobProto blob_proto;
          blob_proto.set_name(name);
          blob_proto.set_type("std::unique_ptr<std::mutex>");
          blob_proto.set_content("");
          acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

///--------------------------------
pub struct MutexDeserializer {
    base: dyn BlobDeserializerBase,
}

impl MutexDeserializer {

    #[inline] pub fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
          *blob->GetMutable<std::unique_ptr<std::mutex>>() =
              std::make_unique<std::mutex>();
        */
    }
}

register_cpu_operator!{Iter, IterOp<CPUContext>}

register_cpu_operator!{AtomicIter, AtomicIterOp<CPUContext>}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{AtomicIter, IDEEPFallbackOp<AtomicIterOp<CPUContext>>}

register_blob_serializer!{
    /*
    (TypeMeta::Id<std::unique_ptr<std::mutex>>()), 
    MutexSerializer
    */
}

register_blob_deserializer!{
    /*
    std::unique_ptr<std::mutex>, 
    MutexDeserializer
    */
}

register_cuda_operator!{Iter, IterOp<CUDAContext>}

register_cuda_operator!{AtomicIter, AtomicIterOp<CUDAContext>}
