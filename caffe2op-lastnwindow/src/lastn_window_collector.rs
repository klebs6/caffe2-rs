crate::ix!();

/**
 | Collect the last N rows from input data. The
 | purpose is to keep track of data accross batches,
 | so for example suppose the LastNWindowCollector is
 | called successively with the following input data
 |
 |   [1, 2, 3, 4]
 |   [5, 6, 7]
 |   [8, 9, 10, 11]
 |
 | And the number of items is set to 6, then the
 | output after the 3rd call will contain the
 | following elements:
 |
 |   [6, 7, 8, 9, 10, 11]
 |
 | No guarantee is made on the ordering of elements
 | in input. 
 |
 | So a valid value for output could have been
 |
 |   [11, 10, 9, 8, 7, 6]
 |
 | Also, this method works for any order tensor,
 | treating the first dimension as input rows and
 | keeping the last N rows seen as input. So for
 | instance:
 |
 |   [[1, 2], [2, 3], [3, 4], [4, 5]]
 |   [[5, 6], [6, 7], [7, 8]]
 |   [[8, 9], [9, 10], [10, 11], [11, 12]]
 |
 | A possible output would be
 |
 |   [[6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]
 |
 | This is not thread safe unless a mutex is given.
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LastNWindowCollectorOp<Context> {
    storage:        OperatorStorage,
    context:        Context,
    num_to_collect: i32,
}

num_inputs!{LastNWindowCollector, (3,4,5)}

num_outputs!{LastNWindowCollector, (2,3)}

inputs!{LastNWindowCollector, 
    0 => ("last-N buffer", "The buffer for last-N record. Should be initialized to empty tensor"),
    1 => ("next cursor",   "The cursor pointing to the next position that should be replaced. Should be initialized to 0."),
    2 => ("DATA",          "tensor to collect from"),
    3 => ("MUTEX",         "(optional) mutex to use to make this thread-safe"),
    4 => ("NUM_VISITED",   "")
}

outputs!{LastNWindowCollector, 
    0 => ("last-N buffer", "Data stored in sessions"),
    1 => ("next cursor",   "Updated input cursor"),
    2 => ("NUM_VISITED",   "number of records seen so far")
}

args!{LastNWindowCollector, 
    0 => ("num_to_collect", "The number of random samples to append for each positive samples")
}

enforce_inplace!{LastNWindowCollector, vec![(0, 0), (1, 1), (4, 2)]}

input_tags!{
    LastNWindowCollectorOp {
        LastNIn,
        NextIn,
        Data,
        Mutex,
        NumVisitedIn
    }
}

output_tags!{
    LastNWindowCollectorOp {
        LastN,
        Next,
        NumVisited
    }
}

impl<Context> LastNWindowCollectorOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            numToCollect_( OperatorStorage::GetSingleArgument<int>("num_to_collect", -1)) 

        CAFFE_ENFORCE_GT(numToCollect_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
        if (InputSize() > MUTEX) {
          auto& mutex = OperatorStorage::Input<std::unique_ptr<std::mutex>>(MUTEX);
          std::lock_guard<std::mutex> guard(*mutex);
          return collect();
        } else {
          return collect();
        }
        */
    }
}

register_cpu_operator!{
    LastNWindowCollector, 
    LastNWindowCollectorOp<CPUContext>
}

should_not_do_gradient!{LastNWindowCollector}
