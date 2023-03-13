crate::ix!();

/**
  | PrefetchOperator is an operator that
  | prefetches the next batch. It should
  | almost always be used to read things
  | from disk, so I am setting the input to
  | zero blobs.
  | 
  | For any operator that is derived from
  | PrefetchOperator, it should explicitly
  | call the Finalize() function in its
  | destructor, so that the prefetching
  | thread is properly destructed.
  | 
  | -----------
  | @note
  | 
  | We inherit from OperatorStorage since
  | we control the synchronization properties
  | of this operator ourselves (we inform
  | the waiting producer after we synchronize).
  | This is a special-case
  | 
  | - you should generally inherit from
  | Operator<Context> directly.
  |
  */
pub struct PrefetchOperator<Context> {
    base:                     OperatorStorage,
    context:                  Context,
    prefetch_access_mutex:    parking_lot::RawMutex,

    /**
      | prefetched_ is used to tell the operator
      | that it is done.
      |
      */
    prefetched:               AtomicBool,

    /**
      | prefetch_success_ is used to see if
      | prefetching failed or not.
      |
      */
    prefetch_success:         AtomicBool,

    producer:                 std::sync::Condvar,
    consumer:                 Condvar,

    /**
      | finalize_ is used to tell the prefetcher
      | to quit.
      |
      */
    finalize:                 AtomicBool,

    prefetch_thread:          Box<Thread>,

    /**
      | Whether to do prefetching or run this
      | as a normal operator
      |
      */
    no_prefetch:              bool,
}

pub trait PrefetchOperatorTrait {

    // You will need to implement this instead of the Run function.
    fn prefetch() -> bool;
    fn copy_prefetched() -> bool;
}

