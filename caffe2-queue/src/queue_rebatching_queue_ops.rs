crate::ix!();

pub type RebatchingQueuePtr = Box<RebatchingQueue>;

caffe_known_type!{RebatchingQueuePtr}

/**
  | Creates the Queue.
  |
  */
pub struct CreateRebatchingQueueOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{CreateRebatchingQueue, CreateRebatchingQueueOp}

num_inputs!{CreateRebatchingQueue, 0}

num_outputs!{CreateRebatchingQueue, 1}

outputs!{CreateRebatchingQueue, 
    0 => ("queue", "object representing the queue")
}

args!{CreateRebatchingQueue, 
    0 => ("num_blobs", "Number of input tensors the queue will support"),
    1 => ("capacity",  "Maximal number of elements the queue can hold at any given point")
}

no_gradient!{CreateRebatchingQueue}

impl CreateRebatchingQueueOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<RebatchingQueuePtr>(0) =
            RebatchingQueuePtr(new RebatchingQueue(
                OperatorStorage::GetSingleArgument<int>("capacity", 1),
                OperatorStorage::GetSingleArgument<int>("num_blobs", 1)));
        return true;
        */
    }
}

/**
  | Enqueues Tensors into the queue.
  | 
  | Number of input tensors should be equal
  | to the number of components passed during
  | creation of the queue.
  | 
  | If the Queue is closed this operation
  | will fail.
  | 
  | If enqueue_batch argument is set. We
  | will split the input tensors by the first
  | dimension to produce single queue elements.
  |
  */
pub struct EnqueueRebatchingQueueOp {

    storage: OperatorStorage,
    context: CPUContext,

    enqueue_batch:  bool,
}

register_cpu_operator!{EnqueueRebatchingQueue, EnqueueRebatchingQueueOp}

num_inputs!{EnqueueRebatchingQueue, (2,INT_MAX)}

num_outputs!{EnqueueRebatchingQueue, 0}

inputs!{EnqueueRebatchingQueue, 
    0 => ("queue", "object representing the queue"),
    1 => ("tensor", "First tensor to enque. ")
}

args!{EnqueueRebatchingQueue, 
    0 => ("enqueue_batch", "Are we enqueuing a batch or just a single element. By default we enqueue single element.")
}

no_gradient!{EnqueueRebatchingQueue}

impl EnqueueRebatchingQueueOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            enqueueBatch_( OperatorStorage::GetSingleArgument<bool>("enqueue_batch", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& queue = Inputs()[0]->template Get<RebatchingQueuePtr>();
        CHECK(queue);
        CAFFE_ENFORCE_EQ(InputSize(), queue->numBlobs() + 1);
        std::vector<const Tensor*> inputTensors;
        inputTensors.reserve(InputSize() - 1);
        for (int i = 1; i < InputSize(); ++i) {
          inputTensors.push_back(&Input(i));
        }

        return enqueueBatch_ ? queue->enqueueMany(context_, inputTensors)
                             : queue->enqueueOne(context_, inputTensors);
        */
    }
}

/**
  | Dequeue Tensors from the Queue.
  | 
  | If the Queue is closed this might return
  | less elements than asked.
  | 
  | If num_elements > 1 the returned elements
  | will be concatenated into one tensor
  | per component.
  |
  */
pub struct DequeueRebatchingQueueOp {
    storage: OperatorStorage,
    context: CPUContext,
    num_elements:  i32,
}

register_cpu_operator!{DequeueRebatchingQueue, DequeueRebatchingQueueOp}

num_inputs!{DequeueRebatchingQueue, 1}

num_outputs!{DequeueRebatchingQueue, (1,INT_MAX)}

inputs!{DequeueRebatchingQueue, 
    0 => ("rebatching_queue", "object representing the queue"),
    1 => ("tensor",           "First tensor to enqueue")
}

args!{DequeueRebatchingQueue, 
    0 => ("num_elements", "Number of elements to dequeue. By default we dequeue one element.")
}

no_gradient!{DequeueRebatchingQueue}

impl DequeueRebatchingQueueOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            numElements_(OperatorStorage::GetSingleArgument<int>("num_elements", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& queue = Inputs()[0]->template Get<RebatchingQueuePtr>();
        CHECK(queue);

        std::vector<Tensor*> outputTensors;
        outputTensors.reserve(OutputSize());
        for (int i = 0; i < OutputSize(); ++i) {
          outputTensors.push_back(Output(i));
        }

        return queue->dequeue(context_, numElements_, outputTensors);
        */
    }
}

///Closes the Queue.
pub struct CloseRebatchingQueueOp {
    base:    OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{CloseRebatchingQueue, CloseRebatchingQueueOp}

num_inputs!{CloseRebatchingQueue, 1}

num_outputs!{CloseRebatchingQueue, 0}

inputs!{CloseRebatchingQueue, 
    0 => ("queue", "object representing the queue")
}

no_gradient!{CloseRebatchingQueue}

impl CloseRebatchingQueueOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize(), 1);
        auto& queue = Inputs()[0]->template Get<RebatchingQueuePtr>();
        CAFFE_ENFORCE(queue);
        queue->close();
        return true;
        */
    }
}
