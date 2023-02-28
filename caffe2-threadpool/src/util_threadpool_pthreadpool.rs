crate::ix!();

/**
  | This is a hack.
  | 
  | Mainly introduced here because
  | 
  | 1. NNPACK can be compiled to use internal
  | legacy threadpool implementation
  | because much of
  | 
  | C2 depends on that.
  | 
  | 2. Then if we want to use NNPACK in PyTorch,
  | which uses new pthreadpool, then we
  | will supply new pthreadpool pointer
  | to NNPACK.
  | 
  | This will not work if NNPACK is compiled
  | with internal legacy threadpool.
  | 
  | Thus this guard along with changes in
  | pthreadpool_impl.cc allows us to override
  | that behavior.
  | 
  | It enables us to use NNPACK from pytorch
  | using `caffe2::pthreadpool_()`
  |
  */
#[cfg(use_pthreadpool)]
pub struct WithCastToNewThreadPool {
    use_new_threadpool:  bool,
}

#[cfg(use_pthreadpool)]
thread_local! {
    pub static using_new_threadpool: AtomicBool = AtomicBool::new(false);
}

#[cfg(use_pthreadpool)]
impl WithCastToNewThreadPool {
    
    #[cfg(use_pthreadpool)]
    pub fn new(use_new_threadpool: bool) -> Self {
        todo!();
        /*
            use_new_threadpool_ = using_new_threadpool;
      using_new_threadpool = use_new_threadpool;
        */
    }
}


#[cfg(use_pthreadpool)]
impl Drop for WithCastToNewThreadPool {
    fn drop(&mut self) {
        todo!();
        /* 
          using_new_threadpool = use_new_threadpool_;
        */
    }
}

pub type legacy_pthreadpool_t                   = *mut threadpool::ThreadPool;//pthreadpool_t
pub type legacy_pthreadpool_function_1d_t       = fn(*mut c_void,usize);
pub type legacy_pthreadpool_function_1d_tiled_t = fn(*mut c_void,usize,usize);
pub type legacy_pthreadpool_function_2d_t       = fn(*mut c_void,usize,usize);
pub type legacy_pthreadpool_function_2d_tiled_t = fn(*mut c_void,usize,usize,usize,usize);
pub type legacy_pthreadpool_function_3d_tiled_t = fn(*mut c_void,usize,usize,usize,usize,usize,usize);
pub type legacy_pthreadpool_function_4d_tiled_t = fn(*mut c_void,usize,usize,usize,usize,usize,usize,usize,usize);

/**
  | Creates a thread pool with the specified
  | number of threads.
  | 
  | -----------
  | @param[in] threads_count
  | 
  | The number of threads in the thread pool.
  | A value of 0 has special interpretation:
  | it creates a thread for each processor
  | core available in the system.
  | 
  | 
  | -----------
  | @return
  | 
  | A pointer to an opaque thread pool object.
  | On error the function returns NULL and
  | sets errno accordingly.
  |
  */

// Returns internal threadpool impl.
#[inline] pub fn legacy_pthreadpool_create(threads_count: usize) -> legacy_pthreadpool_t {
    
    todo!();
    /*
    
    */
}

/**
  | Queries the number of threads in a thread
  | pool.
  | 
  | -----------
  | @param[in] threadpool
  | 
  | The thread pool to query.
  | 
  | 
  | -----------
  | @return
  | 
  | The number of threads in the thread pool.
  |
  */
#[inline] pub fn legacy_pthreadpool_get_threads_count(threadpool: legacy_pthreadpool_t) -> usize {
    
    todo!();
    /*
    
    */
}

/**
  | Processes items in parallel using threads
  | from a thread pool.
  | 
  | When the call returns, all items have
  | been processed and the thread pool is
  | ready for a new task.
  | 
  | -----------
  | @note
  | 
  | If multiple threads call this function
  | with the same thread pool, the calls
  | are serialized.
  | 
  | -----------
  | @param[in] threadpool
  | 
  | The thread pool to use for parallelisation.
  | ----------
  | @param[in] function
  | 
  | The function to call for each item.
  | ----------
  | @param[in] argument
  | 
  | The first argument passed to the @a function.
  | ----------
  | @param[in] items
  | 
  | The number of items to process. The @a
  | function will be called once for each
  | item.
  |
  */
#[inline] pub fn legacy_pthreadpool_compute_1d(
    threadpool: legacy_pthreadpool_t,
    function:   legacy_pthreadpool_function_1d_t,
    argument:   *mut c_void,
    range:      usize)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn legacy_pthreadpool_parallelize_1d(
    threadpool: legacy_pthreadpool_t,
    function:   legacy_pthreadpool_function_1d_t,
    argument:   *mut c_void,
    range:      usize,
    flags:      u32)  {
    
    todo!();
    /*
    
    */
}

/**
  | Terminates threads in the thread pool
  | and releases associated resources.
  | 
  | -----------
  | @param[in,out] threadpool
  | 
  | The thread pool to destroy.
  | 
  | -----------
  | @warning
  | 
  | Accessing the thread pool after a call
  | to this function constitutes undefined
  | behaviour and may cause data corruption.
  |
  */
#[inline] pub fn legacy_pthreadpool_destroy(threadpool: legacy_pthreadpool_t)  {
    
    todo!();
    /*
    
    */
}

/*
pub type pthreadpool_t                   = legacy_pthreadpool_t;
pub type pthreadpool_function_1d_t       = legacy_pthreadpool_function_1d_t;
pub type pthreadpool_function_1d_tiled_t = legacy_pthreadpool_function_1d_tiled_t;
pub type pthreadpool_function_2d_t       = legacy_pthreadpool_function_2d_t;
pub type pthreadpool_function_2d_tiled_t = legacy_pthreadpool_function_2d_tiled_t;
pub type pthreadpool_function_3d_tiled_t = legacy_pthreadpool_function_3d_tiled_t;
pub type pthreadpool_function_4d_tiled_t = legacy_pthreadpool_function_4d_tiled_t;
pub type pthreadpool_create              = legacy_pthreadpool_create;
pub type pthreadpool_destroy             = legacy_pthreadpool_destroy;
pub type pthreadpool_get_threads_count   = legacy_pthreadpool_get_threads_count;
pub type pthreadpool_compute_1d          = legacy_pthreadpool_compute_1d;
pub type pthreadpool_parallelize_1d      = legacy_pthreadpool_parallelize_1d;
pub type pthreadpool_compute_1d_tiled    = legacy_pthreadpool_compute_1d_tiled;
pub type pthreadpool_compute_2d          = legacy_pthreadpool_compute_2d;
pub type pthreadpool_compute_2d_tiled    = legacy_pthreadpool_compute_2d_tiled;
pub type pthreadpool_compute_3d_tiled    = legacy_pthreadpool_compute_3d_tiled;
pub type pthreadpool_compute_4d_tiled    = legacy_pthreadpool_compute_4d_tiled;
*/

#[inline] pub fn divide_round_up(dividend: usize, divisor: usize) -> usize {
    
    todo!();
    /*
        if (dividend % divisor == 0) {
        return dividend / divisor;
      } else {
        return dividend / divisor + 1;
      }
    */
}

#[inline] pub fn min(a: usize, b: usize) -> usize {
    
    todo!();
    /*
        return a < b ? a : b;
    */
}

pub struct compute_1d_tiled_context {
    function: legacy_pthreadpool_function_1d_tiled_t,
    argument: *mut c_void,
    range:    usize,
    tile:     usize,
}

#[inline] pub fn compute_1d_tiled(context: *mut c_void, linear_index: usize)  {
    
    todo!();
    /*
        const struct compute_1d_tiled_context* context = (compute_1d_tiled_context*) context_;
      const size_t tile_index = linear_index;
      const size_t index = tile_index * context->tile;
      const size_t tile = min(context->tile, context->range - index);
      context->function(context->argument, index, tile);
    */
}

#[inline] pub fn legacy_pthreadpool_compute_1d_tiled(
    threadpool: legacy_pthreadpool_t,
    function:   legacy_pthreadpool_function_1d_tiled_t,
    argument:   *mut c_void,
    range:      usize,
    tile:       usize)  
{
    todo!();
    /*
        if (threadpool == NULL) {
        /* No thread pool provided: execute function sequentially on the calling thread */
        for (size_t i = 0; i < range; i += tile) {
          function(argument, i, min(range - i, tile));
        }
      } else {
        /* Execute in parallel on the thread pool using linearized index */
        const size_t tile_range = divide_round_up(range, tile);
        struct compute_1d_tiled_context context = {/*.function = */ function,
                                                   /*.argument = */ argument,
                                                   /*.range = */ range,
                                                   /*.tile = */ tile};
        legacy_pthreadpool_compute_1d(threadpool, (legacy_pthreadpool_function_1d_t) compute_1d_tiled, &context, tile_range);
      }
    */
}

pub struct compute_2d_context {
    function: legacy_pthreadpool_function_2d_t,
    argument: *mut c_void,
    range_j:  FixedDivisor<i32>,
}

#[inline] pub fn compute_2d(
    context: *mut c_void,
    linear_index: usize)
{
    todo!();
    /*
        DCHECK_LE(linear_index, int32_t::max);

      const struct compute_2d_context* context = static_cast<compute_2d_context*>(context_);
      int32_t q;
      int32_t r;
      context->range_j.DivMod(static_cast<int32_t>(linear_index), &q, &r);
      context->function(context->argument, q, r);
    */
}

#[inline] pub fn legacy_pthreadpool_compute_2d(
    threadpool: legacy_pthreadpool_t,
    function:   legacy_pthreadpool_function_2d_t,
    argument:   *mut c_void,
    range_i:    usize,
    range_j:    usize)  
{
    todo!();
    /*
        if (threadpool == NULL) {
        /* No thread pool provided: execute function sequentially on the calling thread */
        for (size_t i = 0; i < range_i; i++) {
          for (size_t j = 0; j < range_j; j++) {
            function(argument, i, j);
          }
        }
      } else {
        DCHECK_LE(range_i * range_j, (size_t)int32_t::max);
        /* Execute in parallel on the thread pool using linearized index */
        struct compute_2d_context context = {
            /*.function = */ function,
            /*.argument = */ argument,
            /*.range_j = */ caffe2::FixedDivisor<int32_t>(range_j)};
        legacy_pthreadpool_compute_1d(threadpool, (legacy_pthreadpool_function_1d_t) compute_2d, &context, range_i * range_j);
      }
    */
}

pub struct compute_2d_tiled_context {
    function:      legacy_pthreadpool_function_2d_tiled_t,
    argument:      *mut c_void,
    tile_range_j:  FixedDivisor<i32>,
    range_i:       usize,
    range_j:       usize,
    tile_i:        usize,
    tile_j:        usize,
}

#[inline] pub fn compute_2d_tiled(
    context: *mut c_void,
    linear_index: usize)  
{
    todo!();
    /*
        int32_t q;
      int32_t r;

      const struct compute_2d_tiled_context* context = static_cast<compute_2d_tiled_context*>(context_);
      context->tile_range_j.DivMod(linear_index, &q, &r);
      const size_t max_tile_i = context->tile_i;
      const size_t max_tile_j = context->tile_j;
      const size_t index_i = q * max_tile_i;
      const size_t index_j = r * max_tile_j;
      const size_t tile_i = min(max_tile_i, context->range_i - index_i);
      const size_t tile_j = min(max_tile_j, context->range_j - index_j);
      context->function(context->argument, index_i, index_j, tile_i, tile_j);
    */
}

#[inline] pub fn legacy_pthreadpool_compute_2d_tiled(
    threadpool:  legacy_pthreadpool_t,
    function:    legacy_pthreadpool_function_2d_tiled_t,
    argument:    *mut c_void,
    range_i:     usize,
    range_j:     usize,
    tile_i:      usize,
    tile_j:      usize)  
{

    todo!();
    /*
        if (threadpool == NULL) {
        /* No thread pool provided: execute function sequentially on the calling thread */
        for (size_t i = 0; i < range_i; i += tile_i) {
          for (size_t j = 0; j < range_j; j += tile_j) {
            function(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j));
          }
        }
      } else {
        /* Execute in parallel on the thread pool using linearized index */
        const size_t tile_range_i = divide_round_up(range_i, tile_i);
        const size_t tile_range_j = divide_round_up(range_j, tile_j);
        DCHECK_LE(
            tile_range_i * tile_range_j,
            (size_t)int32_t::max);
        struct compute_2d_tiled_context context = {
            /*.function = */ function,
            /*.argument = */ argument,
            /*.tile_range_j = */ caffe2::FixedDivisor<int32_t>(tile_range_j),
            /*.range_i = */ range_i,
            /*.range_j = */ range_j,
            /*.tile_i = */ tile_i,
            /*.tile_j = */ tile_j};
        legacy_pthreadpool_compute_1d(threadpool, (legacy_pthreadpool_function_1d_t) compute_2d_tiled, &context, tile_range_i * tile_range_j);
      }
    */
}

pub struct compute_3d_tiled_context {
    function:      legacy_pthreadpool_function_3d_tiled_t,
    argument:      *mut c_void,
    tile_range_j:  FixedDivisor<i32>,
    tile_range_k:  FixedDivisor<i32>,
    range_i:       usize,
    range_j:       usize,
    range_k:       usize,
    tile_i:        usize,
    tile_j:        usize,
    tile_k:        usize,
}

#[inline] pub fn compute_3d_tiled(
    context: *mut c_void,
    linear_index: usize)  
{
    todo!();
    /*
        int32_t tile_index_ij, tile_index_k;
      const struct compute_3d_tiled_context* context = static_cast<compute_3d_tiled_context*>(context_);
      context->tile_range_k.DivMod(
          static_cast<int32_t>(linear_index), &tile_index_ij, &tile_index_k);
      int32_t tile_index_i, tile_index_j;
      context->tile_range_j.DivMod(tile_index_ij, &tile_index_i, &tile_index_j);
      const size_t max_tile_i = context->tile_i;
      const size_t max_tile_j = context->tile_j;
      const size_t max_tile_k = context->tile_k;
      const size_t index_i = static_cast<uint32_t>(tile_index_i) * max_tile_i;
      const size_t index_j = static_cast<uint32_t>(tile_index_j) * max_tile_j;
      const size_t index_k = static_cast<uint32_t>(tile_index_k) * max_tile_k;
      const size_t tile_i = min(max_tile_i, context->range_i - index_i);
      const size_t tile_j = min(max_tile_j, context->range_j - index_j);
      const size_t tile_k = min(max_tile_k, context->range_k - index_k);
      context->function(
          context->argument, index_i, index_j, index_k, tile_i, tile_j, tile_k);
    */
}


#[inline] pub fn legacy_pthreadpool_compute_3d_tiled(
    threadpool:  legacy_pthreadpool_t,
    function:    legacy_pthreadpool_function_3d_tiled_t,
    argument:    *mut c_void,
    range_i:     usize,
    range_j:     usize,
    range_k:     usize,
    tile_i:      usize,
    tile_j:      usize,
    tile_k:      usize)  
{
    todo!();
    /*
        if (threadpool == NULL) {
        /* No thread pool provided: execute function sequentially on the calling
         * thread */
        for (size_t i = 0; i < range_i; i += tile_i) {
          for (size_t j = 0; j < range_j; j += tile_j) {
            for (size_t k = 0; k < range_k; k += tile_k) {
              function(
                  argument,
                  i,
                  j,
                  k,
                  min(range_i - i, tile_i),
                  min(range_j - j, tile_j),
                  min(range_k - k, tile_k));
            }
          }
        }
      } else {
        /* Execute in parallel on the thread pool using linearized index */
        const size_t tile_range_i = divide_round_up(range_i, tile_i);
        const size_t tile_range_j = divide_round_up(range_j, tile_j);
        const size_t tile_range_k = divide_round_up(range_k, tile_k);
        DCHECK_LE(
            tile_range_i * tile_range_j * tile_range_k,
            (size_t)int::max);
        struct compute_3d_tiled_context context = {
            /*.function = */ function,
            /*.argument = */ argument,
            /*.tile_range_j = */ caffe2::FixedDivisor<int>(tile_range_j),
            /*.tile_range_k = */ caffe2::FixedDivisor<int>(tile_range_k),
            /*.range_i = */ range_i,
            /*.range_j = */ range_j,
            /*.range_k = */ range_k,
            /*.tile_i = */ tile_i,
            /*.tile_j = */ tile_j,
            /*.tile_k = */ tile_k};
        legacy_pthreadpool_compute_1d(
            threadpool,
            (legacy_pthreadpool_function_1d_t)compute_3d_tiled,
            &context,
            tile_range_i * tile_range_j * tile_range_k);
      }
    */
}

pub struct compute_4d_tiled_context {

    function:       legacy_pthreadpool_function_4d_tiled_t,
    argument:       *mut c_void,
    tile_range_kl:  FixedDivisor<i32>,
    tile_range_j:   FixedDivisor<i32>,
    tile_range_l:   FixedDivisor<i32>,
    range_i:        usize,
    range_j:        usize,
    range_k:        usize,
    range_l:        usize,
    tile_i:         usize,
    tile_j:         usize,
    tile_k:         usize,
    tile_l:         usize,
}

#[inline] pub fn compute_4d_tiled(
    context: *mut c_void,
    linear_index: usize)  
{
    todo!();
    /*
        int32_t tile_index_ij, tile_index_kl;
      const struct compute_4d_tiled_context* context = static_cast<compute_4d_tiled_context*>(context_);
      context->tile_range_kl.DivMod(
          static_cast<int32_t>(linear_index), &tile_index_ij, &tile_index_kl);
      int32_t tile_index_i, tile_index_j;
      context->tile_range_j.DivMod(tile_index_ij, &tile_index_i, &tile_index_j);
      int32_t tile_index_k, tile_index_l;
      context->tile_range_l.DivMod(tile_index_kl, &tile_index_k, &tile_index_l);
      const size_t max_tile_i = context->tile_i;
      const size_t max_tile_j = context->tile_j;
      const size_t max_tile_k = context->tile_k;
      const size_t max_tile_l = context->tile_l;
      const size_t index_i = static_cast<uint32_t>(tile_index_i) * max_tile_i;
      const size_t index_j = static_cast<uint32_t>(tile_index_j) * max_tile_j;
      const size_t index_k = static_cast<uint32_t>(tile_index_k) * max_tile_k;
      const size_t index_l = static_cast<uint32_t>(tile_index_l) * max_tile_l;
      const size_t tile_i = min(max_tile_i, context->range_i - index_i);
      const size_t tile_j = min(max_tile_j, context->range_j - index_j);
      const size_t tile_k = min(max_tile_k, context->range_k - index_k);
      const size_t tile_l = min(max_tile_l, context->range_l - index_l);
      context->function(
          context->argument,
          index_i,
          index_j,
          index_k,
          index_l,
          tile_i,
          tile_j,
          tile_k,
          tile_l);
    */
}

#[inline] pub fn legacy_pthreadpool_compute_4d_tiled(
    threadpool:  legacy_pthreadpool_t,
    function:    legacy_pthreadpool_function_4d_tiled_t,
    argument:    *mut c_void,
    range_i:     usize,
    range_j:     usize,
    range_k:     usize,
    range_l:     usize,
    tile_i:      usize,
    tile_j:      usize,
    tile_k:      usize,
    tile_l:      usize)  
{
    todo!();
    /*
        if (threadpool == NULL) {
        /* No thread pool provided: execute function sequentially on the calling
         * thread */
        for (size_t i = 0; i < range_i; i += tile_i) {
          for (size_t j = 0; j < range_j; j += tile_j) {
            for (size_t k = 0; k < range_k; k += tile_k) {
              for (size_t l = 0; l < range_l; l += tile_l) {
                function(
                    argument,
                    i,
                    j,
                    k,
                    l,
                    min(range_i - i, tile_i),
                    min(range_j - j, tile_j),
                    min(range_k - k, tile_k),
                    min(range_l - l, tile_l));
              }
            }
          }
        }
      } else {
        /* Execute in parallel on the thread pool using linearized index */
        const size_t tile_range_i = divide_round_up(range_i, tile_i);
        const size_t tile_range_j = divide_round_up(range_j, tile_j);
        const size_t tile_range_k = divide_round_up(range_k, tile_k);
        const size_t tile_range_l = divide_round_up(range_l, tile_l);
        DCHECK_LE(
            tile_range_i * tile_range_j * tile_range_k * tile_range_l,
            (size_t)int::max);
        struct compute_4d_tiled_context context = {
            /*.function = */ function,
            /*.argument = */ argument,
            /*.tile_range_kl = */
            caffe2::FixedDivisor<int>(tile_range_k * tile_range_l),
            /*.tile_range_j = */ caffe2::FixedDivisor<int>(tile_range_j),
            /*.tile_range_l = */ caffe2::FixedDivisor<int>(tile_range_l),
            /*.range_i = */ range_i,
            /*.range_j = */ range_j,
            /*.range_k = */ range_k,
            /*.range_l = */ range_l,
            /*.tile_i = */ tile_i,
            /*.tile_j = */ tile_j,
            /*.tile_k = */ tile_k,
            /*.tile_l = */ tile_l};
        legacy_pthreadpool_compute_1d(
            threadpool,
            (legacy_pthreadpool_function_1d_t)compute_4d_tiled,
            &context,
            tile_range_i * tile_range_j * tile_range_k * tile_range_l);
      }
    */
}
