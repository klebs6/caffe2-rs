crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Parallel.h]

#[inline] pub fn divup(x: i64, y: i64) -> i64 {
    
    todo!();
        /*
            return (x + y - 1) / y;
        */
}

/**
  | Called during new thread initialization
  |
  */
pub fn init_num_threads()  {
    
    todo!();
        /*
        
        */
}

/**
  | Sets the number of threads to be used
  | in parallel region
  |
  */
pub fn set_num_threads(_0: i32)  {
    
    todo!();
        /*
        
        */
}

/**
  | Returns the maximum number of threads
  | that may be used in a parallel region
  |
  */
pub fn get_num_threads() -> i32 {
    
    todo!();
        /*
        
        */
}

/**
  | Returns the current thread number (starting
  | from 0) in the current parallel region, or 0 in
  | the sequential region
  |
  */
pub fn get_thread_num() -> i32 {
    
    todo!();
        /*
        
        */
}

/**
  | Checks whether the code runs in parallel
  | region
  |
  */
pub fn in_parallel_region() -> bool {
    
    todo!();
        /*
        
        */
}

/**
  | Initialise num_threads lazily at first
  | parallel call
  |
  */
#[inline] pub fn lazy_init_num_threads()  {
    
    todo!();
        /*
            thread_local bool init = false;
      if (C10_UNLIKELY(!init)) {
        init_num_threads();
        init = true;
      }
        */
}

/**
  | parallel_for
  | 
  | begin: index at which to start applying
  | user function
  | 
  | end: index at which to stop applying
  | user function
  | 
  | grain_size: number of elements per
  | chunk. impacts the degree of parallelization
  | 
  | f: user function applied in parallel
  | to the chunks, signature: void f(i64
  | begin, i64 end)
  | 
  | Warning: parallel_for does NOT copy
  | thread local states from the current
  | thread to the worker threads.
  | 
  | This means for example that Tensor operations
  | CANNOT be used in the body of your function,
  | only data pointers.
  |
  */
#[inline] pub fn parallel_for<F>(
        begin:      i64,
        end:        i64,
        grain_size: i64,
        f:          &F)  {
    
    todo!();
        /*
        
        */
}

/**
  | parallel_reduce
  | 
  | begin: index at which to start applying
  | reduction
  | 
  | end: index at which to stop applying
  | reduction
  | 
  | grain_size: number of elements per
  | chunk. impacts number of elements in
  | intermediate results tensor and degree
  | of parallelization.
  | 
  | ident: identity for binary combination
  | function sf. sf(ident, x) needs to return
  | x.
  | 
  | f: function for reduction over a chunk.
  | f needs to be of signature Scalar f(i64
  | partial_begin, i64 partial_end,
  | Scalar identifiy)
  | 
  | sf: function to combine two partial
  | results. sf needs to be of signature
  | Scalar sf(Scalar x, Scalar y)
  | 
  | For example, you might have a tensor
  | of 10000 entires and want to sum together
  | all the elements. Parallel_reduce
  | with a grain_size of 2500 will then allocate
  | an intermediate result tensor with
  | 4 elements. Then it will execute the
  | function "f" you provide and pass the
  | beginning and end index of these chunks,
  | so 0-2499, 2500-4999, etc. and the combination
  | identity. It will then write out the
  | result from each of these chunks into
  | the intermediate result tensor. After
  | that it'll reduce the partial results
  | from each chunk into a single number
  | using the combination function sf and
  | the identity ident. For a total summation
  | this would be "+" and 0 respectively.
  | This is similar to tbb's approach [1],
  | where you need to provide a function
  | to accumulate a subrange, a function
  | to combine two partial results and an
  | identity.
  | 
  | Warning: parallel_reduce does NOT
  | copy thread local states from the current
  | thread to the worker threads.
  | 
  | This means for example that Tensor operations
  | CANNOT be used in the body of your function,
  | only data pointers.
  | 
  | [1] https://software.intel.com/en-us/node/506154
  |
  */
//template <class Scalar, class F, class SF>
#[inline] pub fn parallel_reduce(
        begin:      i64,
        end:        i64,
        grain_size: i64,
        ident:      Scalar,
        f:          &F,
        sf:         &SF) -> Scalar {
    
    todo!();
        /*
        
        */
}

/**
  | Returns a detailed string describing
  | parallelization settings
  |
  */
pub fn get_parallel_info() -> String {
    
    todo!();
        /*
        
        */
}

/**
  | Sets number of threads used for inter-op
  | parallelism
  |
  */
pub fn set_num_interop_threads(_0: i32)  {
    
    todo!();
        /*
        
        */
}

/**
  | Returns the number of threads used for
  | inter-op parallelism
  |
  */
pub fn get_num_interop_threads() -> i32 {
    
    todo!();
        /*
        
        */
}

/**
  | Launches inter-op parallel task
  |
  */
pub fn launch(func: fn() -> ())  {
    
    todo!();
        /*
        
        */
}

pub fn launch_no_thread_state(fn_: fn() -> ())  {
    
    todo!();
        /*
        
        */
}

/// Launches intra-op parallel task
pub fn intraop_launch(func: fn() -> ())  {
    
    todo!();
        /*
        
        */
}

/// Launches intra-op parallel task, returns a future
pub fn intraop_launch_future(func: fn() -> ()) -> IntrusivePtr<Future> {
    
    todo!();
        /*
        
        */
}

/// Returns number of intra-op threads used by default
pub fn intraop_default_num_threads() -> i32 {
    
    todo!();
        /*
        
        */
}
