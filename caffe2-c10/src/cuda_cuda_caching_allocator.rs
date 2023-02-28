crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDACachingAllocator.h]

#[cfg(feature = "cuda")]
pub struct CUDAOutOfMemoryError {
    base: Error,
}

/**
  | Caching allocator will execute every registered
  | callback if it unable to find block inside of
  | already allocated area.
  |
  */
#[cfg(feature = "cuda")]
pub trait FreeMemoryCallbackInterface: Execute {}

pub trait Execute {

    fn execute(&mut self) -> bool;
}

c10_declare_registry!{
    FreeCudaMemoryCallbacksRegistry, 
    FreeMemoryCallback,
}

macro_rules! register_free_memory_callback {
    ($name:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_CLASS(FreeCudaMemoryCallbacksRegistry, name, __VA_ARGS__);
        */
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {

    use super::*;

    /**
      | Yet another caching allocator for Cuda
      | device allocations.
      |
      | - Allocations are associated with
      |   a stream. Once freed, blocks can be
      |   re-allocated on the same stream, but not
      |   on any other stream.
      |
      | - The allocator attempts to find the
      |   smallest cached block that will fit the
      |   requested size. If the block is larger
      |   than the requested size, it may be
      |   split. If no block is found, the
      |   allocator will delegate to cudaMalloc.
      |
      | - If the cudaMalloc fails, the allocator
      |   will free all cached blocks that are not
      |   split and retry the allocation.
      |
      | - Large (>1MB) and small allocations are
      |   stored in separate pools. Small requests
      |   are packed into 2MB buffers. Large
      |   requests will use the smallest available
      |   free block or allocate a new block using
      |   cudaMalloc.
      |
      |   To reduce fragmentation, requests between
      |   1MB and 10MB will allocate and split
      |   a 20MB block, if no free block of
      |   sufficient size is available.
      |
      | With this allocator, allocations and frees
      | should logically be considered "usages" of
      | the memory segment associated with streams,
      | just like kernel launches. The programmer
      | must insert the proper synchronization if
      | memory segments are used from multiple
      | streams.
      |
      | The library provides a recordStream()
      | function to help insert the correct
      | synchronization when allocations are used
      | on multiple streams. This will ensure that
      | the block is not reused before each
      | recorded stream completes work.
      |
      | TODO: Turn this into an honest to goodness
      | class.
      |
      | I briefly attempted to do this, but it was
      | a bit irritating to figure out how to also
      | correctly apply pimpl pattern so I didn't
      | have to leak any internal implementation
      | details in the header (CUDACachingAllocator
      | could be made a pimpl, but you also need to
      | appropriately define a class which is
      | a subclass of Allocator. Not impossible,
      | but required a bit more surgery than
      | I wanted to do at the time.)
      |
      | Why is this using a namespace rather than
      | old-style THCCachingAllocator_ prefix?
      |
      | Mostly because it made the HIPify rules
      | easier to write; _ is not counted as a word
      | boundary, so you would otherwise have to
      | list each of these functions.
      */
    pub mod CUDACachingAllocator {

        use super::*;

        pub struct Stat {
            current:   i64, // default = 0
            peak:      i64, // default = 0
            allocated: i64, // default = 0
            freed:     i64, // default = 0
        }

        #[repr(u64)]
        pub enum StatType {
            AGGREGATE  = 0,
            SMALL_POOL = 1,
            LARGE_POOL = 2,

            /// remember to update this whenever
            /// a new stat type is added
            ///
            NUM_TYPES  = 3 
        }

        pub type StatArray = Array<Stat,StatType_NUM_TYPES>;

        /**
          | Struct containing memory allocator
          | summary statistics for a device.
          |
          */
        pub struct DeviceStats {

            /**
              | COUNT: allocations requested by client
              | code
              |
              */
            allocation:           StatArray,

            /**
              | COUNT: number of allocated segments
              | from cudaMalloc().
              |
              */
            segment:              StatArray,

            /**
              | COUNT: number of active memory blocks
              | (allocated or used by stream)
              |
              */
            active:               StatArray,

            /**
              | COUNT: number of inactive, split memory
              | blocks (unallocated but can't be released
              | via cudaFree)
              |
              */
            inactive_split:       StatArray,

            /**
              | SUM: bytes requested by client code
              |
              */
            allocated_bytes:      StatArray,

            /**
              | SUM: bytes reserved by this memory allocator
              | (both free and used)
              |
              */
            reserved_bytes:       StatArray,

            /**
              | SUM: bytes within active memory blocks
              |
              */
            active_bytes:         StatArray,

            /**
              | SUM: bytes within inactive, split memory
              | blocks
              |
              */
            inactive_split_bytes: StatArray,

            /**
              | COUNT: total number of failed calls
              | to Cuda malloc necessitating cache
              | flushes.
              |
              */
            num_alloc_retries:    i64, // default = 0

            /**
              | COUNT: total number of OOMs (i.e. failed
              | calls to Cuda after cache flush)
              |
              */
            num_ooms:             i64, // default = 0
        }

        /**
          | Struct containing info of an allocation
          | block (i.e. a fractional part of
          | a cudaMalloc)..
          |
          */
        pub struct BlockInfo {
            size:      i64, // default = 0
            allocated: bool, // default = false
            active:    bool, // default = false
        }

        /**
          | Struct containing info of a memory
          | segment (i.e. one contiguous
          | cudaMalloc).
          |
          */
        pub struct SegmentInfo {
            device:         i64, // default = 0
            address:        i64, // default = 0
            total_size:     i64, // default = 0
            allocated_size: i64, // default = 0
            active_size:    i64, // default = 0
            is_large:       bool, // default = false
            blocks:         Vec<BlockInfo>,
        }

        #[cfg(feature = "cuda")] 
        pub fn raw_alloc(nbytes: usize)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn raw_alloc_with_stream(
            nbytes: usize,
            stream: CudaStream)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn raw_delete(ptr: *mut c_void)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn get() -> *mut Allocator {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn init(device_count: i32)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn set_memory_fraction(
            fraction: f64,
            device:   i32)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn empty_cache()  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn cache_info(
            dev_id:          i32,
            cached_and_free: *mut usize,
            largest_block:   *mut usize)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn get_base_allocation(
            ptr:  *mut c_void,
            size: *mut usize)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn record_stream(
            _0:     &DataPtr,
            stream: CudaStream)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn get_device_stats(device: i32) -> DeviceStats {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn reset_accumulated_stats(device: i32)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn reset_peak_stats(device: i32)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn snapshot() -> Vec<SegmentInfo> {

            todo!();
            /*

            */
        }

        // CUDAGraph interactions
        #[cfg(feature = "cuda")] 
        pub fn notify_capture_begin(
            device:     i32,
            graph_id:   CaptureId,
            mempool_id: MempoolId)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn notify_capture_end(
            device:   i32,
            graph_id: CaptureId)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn notify_capture_destroy(
            device:     i32,
            mempool_id: MempoolId)  {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn get_free_mutex() -> *mut Mutex {

            todo!();
            /*

            */
        }

        #[cfg(feature = "cuda")] 
        pub fn get_ipc_dev_ptr(handle: String) -> Arc<c_void> {

            todo!();
            /*

            */
        }

        /**
         * Note [Interaction with Cuda graph capture]
         * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         *
         * Graph capture performs a dry run of
         * a region of execution, freezing all
         * Cuda work (and virtual addresses used
         * during that work) into a "graph." The
         * graph may be "replayed" like a single
         * giant kernel, with greatly reduced CPU
         * overhead as well as modestly improved
         * GPU performance.
         *
         * Because capture bakes in memory
         * addresses, the memory used during
         * capture must be available for the graph
         * to use during
         * replay. DeviceCachingAllocator assigns
         * and frees memory eagerly and
         * dynamically, so if we're not careful
         * about managing graphs' memory, at
         * replay time those memory addresses
         * could be use by other tensors.
         *
         * To guarantee a graph's baked in
         * addresses are safe to reuse in replay,
         * DeviceAllocator satisfies allocations
         * from a graph-private memory pool during
         * capture, and doesn't begin cudaFreeing
         * those addresses until the graph is
         * destroyed.
         *
         * Within the private pool, allocations
         * are freed and reassigned as usual
         * during capture. Memory regions will be
         * used in a consistent order during
         * replay. So a private pool doesn't use
         * memory more wastefully than the default
         * pools during capture, but it does
         * reserve its high-water mark of used
         * memory away from the default pools as
         * long as the capture(s) it served
         * survive (regardless whether those
         * captures are idle or replaying).
         *
         * CUDAGraph's requests for private pools
         * are mediated by
         * DeviceAllocator::notifyCaptureBegin,
         * notifyCaptureEnd, and
         * notifyCaptureDestroy.
         */
        pub type StreamSet = HashSet<CudaStream>;

        /// all sizes are rounded to at least 512 bytes
        pub const K_MIN_BLOCK_SIZE: usize = 512;

        /// largest "small" allocation is 1 MiB
        pub const K_SMALL_SIZE: usize = 1048576;

        /// "small" allocations are packed in 2 MiB blocks
        pub const K_SMALL_BUFFER: usize = 2097152;

        /// "large" allocations may be packed in 20 MiB blocks
        pub const K_LARGE_BUFFER: usize = 20971520;

        /// allocations between 1 and 10 MiB may use kLargeBuffer
        pub const K_MIN_LARGE_ALLOC: usize = 10485760;

        /// round up large allocations to 2 MiB
        pub const K_ROUND_LARGE: usize = 2097152;

        pub type StatTypes = BitSet<StatType_NUM_TYPES>;

        pub fn update_stat(
                stat:   &mut Stat,
                amount: i64)  {
            
            todo!();
            /*
                stat.current += amount;

                    TORCH_INTERNAL_ASSERT(
                        stat.current >= 0,
                        "Negative tracked stat in Cuda allocator (likely logic error).");

                    stat.peak = max(stat.current, stat.peak);
                    if (amount > 0) {
                        stat.allocated += amount;
                    }
                    if (amount < 0) {
                        stat.freed += -amount;
                    }
            */
        }

        pub fn reset_accumulated_stat(stat: &mut Stat)  {
            
            todo!();
                /*
                    stat.allocated = 0;
                        stat.freed = 0;
                */
        }

        pub fn reset_peak_stat(stat: &mut Stat)  {
            
            todo!();
                /*
                    stat.peak = stat.current;
                */
        }

        pub fn update_stat_array(
            stat_array: &mut StatArray,
            amount:     i64,
            stat_types: &StatTypes)  {
            
            todo!();
                /*
                    for (const auto stat_type : irange(stat_types.size())) {
                            if (stat_types[stat_type]) {
                                update_stat(stat_array[stat_type], amount);
                            }
                        }
                */
        }

        pub type Comparison = fn(_0: *const Block, _1: *const Block) -> bool;

        pub struct BlockPool {
            blocks:             HashSet<*mut Block,Comparison>,
            is_small:           bool,
            owner_private_pool: *mut PrivatePool,
        }

        impl BlockPool {
            
            pub fn new(
                comparator:   Comparison,
                small:        bool,
                private_pool: *mut PrivatePool) -> Self {

                let private_pool: *mut PrivatePool =
                         private_pool.unwrap_or(nullptr);

                todo!();
                /*
                : blocks(comparator),
                : is_small(small),
                : owner_private_pool(private_pool),

                
                */
            }
        }

        // innovation is the child of freedom and
        // the parent of prosperity
        //
        // -unknown
        //
        pub struct Block {

            /**
              | gpu
              |
              */
            device:      i32,


            /**
              | allocation stream
              |
              */
            stream:      CudaStream,


            /**
              | streams on which the block was used
              |
              */
            stream_uses: StreamSet,


            /**
              | block size in bytes
              |
              */
            size:        usize,


            /**
              | owning memory pool
              |
              */
            pool:        *mut BlockPool,


            /**
              | memory address
              |
              */
            ptr:         *mut c_void,


            /**
              | in-use flag
              |
              */
            allocated:   bool,


            /**
              | prev block if split from a larger allocation
              |
              */
            prev:        *mut Block,


            /**
              | next block if split from a larger allocation
              |
              */
            next:        *mut Block,


            /**
              | number of outstanding Cuda events
              |
              */
            event_count: i32,
        }

        impl Block {

            pub fn new(
                device: i32,
                stream: CudaStream,
                size:   usize,
                pool:   *mut BlockPool,
                ptr:    *mut c_void) -> Self {
            
                todo!();
                /*
                : device(device),
                : stream(stream),
                : stream_uses(),
                : size(size),
                : pool(pool),
                : ptr(ptr),
                : allocated(0),
                : prev(nullptr),
                : next(nullptr),
                : event_count(0),
                */
            }

            /// constructor for search key
            pub fn new(
                device: i32,
                stream: CudaStream,
                size:   usize) -> Self {
            
                todo!();
                /*
                : device(device),
                : stream(stream),
                : stream_uses(),
                : size(size),
                : pool(nullptr),
                : ptr(nullptr),
                : allocated(0),
                : prev(nullptr),
                : next(nullptr),
                : event_count(0),
                */
            }
            
            pub fn is_split(&self) -> bool {
                
                todo!();
                /*
                    return (prev != nullptr) || (next != nullptr);
                */
            }
        }

        pub fn block_comparator(
            a: *const Block,
            b: *const Block) -> bool {
            
            todo!();
                /*
                    if (a->stream != b->stream) {
                            return (uintptr_t)a->stream < (uintptr_t)b->stream;
                        }
                        if (a->size != b->size) {
                            return a->size < b->size;
                        }
                        return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
                */
        }

        pub fn format_size(size: u64) -> String {
            
            todo!();
                /*
                    ostringstream os;
                        os.precision(2);
                        os << fixed;
                        if (size <= 1024) {
                            os << size << " bytes";
                        } else if (size <= 1048576) {
                            os << (size / 1024.0);
                            os << " KiB";
                        } else if (size <= 1073741824ULL) {
                            os << size / 1048576.0;
                            os << " MiB";
                        } else {
                            os << size / 1073741824.0;
                            os << " GiB";
                        }
                        return os.str();
                */
        }

        pub struct AllocParams {
            search_key: Block,
            pool:       *mut BlockPool,
            alloc_size: usize,
            block:      *mut Block,
            stat_types: StatTypes,
            err:        CudaError,
        }

        impl AllocParams {
            
            pub fn new(
                device:     i32,
                size:       usize,
                stream:     CudaStream,
                pool:       *mut BlockPool,
                alloc_size: usize,
                stats:      &mut DeviceStats) -> Self {
            
                todo!();
                /*
                : search_key(device, stream, size),
                : pool(pool),
                : alloc_size(alloc_size),
                : block(nullptr),
                : err(cudaSuccess),

                
                */
            }
            
            pub fn device(&mut self) -> i32 {
                
                todo!();
                /*
                    return search_key.device;
                */
            }
            
            
            pub fn stream(&mut self) -> CudaStream {
                
                todo!();
                /*
                    return search_key.stream;
                */
            }
            
            
            pub fn size(&mut self) -> usize {
                
                todo!();
                /*
                    return search_key.size;
                */
            }
        }

        /// Cuda graphs helper
        pub struct PrivatePool {

            /**
              | Number of live graphs using this pool
              |
              */
            use_count:         i32,

            /**
              | Number of unfreed cudaMallocs made
              | for this pool. When use_count and cudaMalloc_count
              | drop to zero, we can delete this PrivatePool
              | from graph_pools.
              |
              */
            cuda_malloc_count: i32,

            /**
              | Instead of maintaining private BlockPools
              | here, I could stuff all blocks (private
              | or no) into the top-level large_blocks
              | and small_blocks, and distinguish
              | private blocks by adding a "pool id"
              | check above the stream check in BlockComparator.
              | BlockComparator is performance- critial
              | though,
              | 
              | I'd rather not add more logic to it.
              |
              */
            large_blocks:      BlockPool,

            small_blocks:      BlockPool,
        }

        impl Default for PrivatePool {
            
            fn default() -> Self {
                todo!();
                /*


                    : use_count(1),
                            cudaMalloc_count(0),
                            large_blocks(BlockComparator, /*is_small=*/false, this),
                            small_blocks(BlockComparator, /*is_small=*/true, this)
                */
            }
        }

        pub struct MempoolIdHash {

        }

        impl MempoolIdHash {
            
            pub fn invoke(&self, mempool_id: &MempoolId) -> usize {
                
                todo!();
                /*
                    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
                */
            }
        }

        pub fn cuda_malloc_maybe_capturing(
                p:    *mut *mut c_void,
                size: usize) -> CudaError {
            
            todo!();
                /*
                    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
                        if (currentStreamCaptureStatusMayInitCtx() ==
                            CaptureStatus::None) {
            #endif
                            return cudaMalloc(p, size);
            #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
                        } else {
                            // It's ok to capture cudaMallocs, as long as we never cudaFree those
                            // addresses before replay.
                            CudaStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
                            return cudaMalloc(p, size);
                        }
            #endif
                */
        }

        pub struct DeviceCachingAllocator {

            /**
              | lock around all operations
              |
              */
            mutex:                                  RefCell<RecursiveMutex>,

            /**
              | device statistics
              |
              */
            stats:                                  DeviceStats,


            /**
              | unallocated cached blocks larger than
              | 1 MB
              |
              */
            large_blocks:                           BlockPool,


            /**
              | unallocated cached blocks 1 MB or smaller
              |
              */
            small_blocks:                           BlockPool,


            /**
              | allocated or in use by a stream. Holds
              | all active allocations, whether they
              | came from graph_pools or one of the BlockPools
              | above.
              |
              */
            active_blocks:                          HashSet<*mut Block>,


            /**
              | captures_underway tracks if a capture
              | might be underway on any stream.
              | 
              | Most of the time it's zero, in which case
              | malloc can avoid calling cudaStreamGetCaptureInfo
              | in the hot path.
              |
              */
            captures_underway:                      i32, // default = 0


            /**
              | See free() for this thing's purpose
              |
              */
            needs_events_deferred_until_no_capture: Vec<*mut Block>,


            /**
              | outstanding cuda events
              |
              */
            cuda_events:                            VecDeque<Pair<cuda::Event,*mut Block>>,


            /**
              | record used memory.
              |
              */
            total_allocated_memory:                 usize, // default = 0

            allowed_memory_maximum:                 usize, // default = 0
            set_fraction:                           bool, // default = false

            /**
              | Members specific to Cuda graphs
              | 
              | Private pools for Cuda graphs
              |
              */
            graph_pools:                            HashMap<MempoolId,Box<PrivatePool>,MempoolIdHash>,


            /**
              | Pools no longer referenced by any graph.
              | Their BlockPools are eligible for free_blocks.
              | Can't be a vector or deque because we
              | might erase entries in any order. Could
              | be an list, but we don't care much, access
              | and insert/erase are rare.
              |
              */
            graph_pools_freeable:                   HashMap<MempoolId,*mut PrivatePool,MempoolIdHash>,


            /**
              | Maps a capturing stream to its assigned
              | private pool, in case we want multiple
              | captures to share the same pool
              |
              */
            capture_to_pool_map:                    HashMap<CaptureId,MempoolId>,
        }

        impl Default for DeviceCachingAllocator {
            
            fn default() -> Self {
                todo!();
                /*


                    : large_blocks(BlockComparator, /*is_small=*/false),
                            small_blocks(BlockComparator, /*is_small=*/true)
                */
            }
        }

        impl DeviceCachingAllocator {

            /**
              | All public methods (except the
              | above) acquire the allocator mutex.
              |
              | Thus, do not call a public method
              | from another public method.
              |
              */
            pub fn malloc(&mut self, 
                device: i32,
                size:   usize,
                stream: CudaStream) -> *mut Block {
                
                todo!();
                /*
                    unique_lock<recursive_mutex> lock(mutex);

                            if (C10_LIKELY(captures_underway == 0)) {
                                // Processes end-of-life events for outstanding allocations used on
                                // multiple streams (checks if their GPU-side uses are complete and
                                // recycles their memory if so)
                                //
                                // Q. Why skip process_events if a capture might be underway?
                                // A. process_events involves cudaEventQueries, illegal during Cuda graph
                                // capture.
                                //    Dumb simple solution: defer reclaiming these allocations until after
                                //    capture. Cross-stream memory use is uncommon, so the deferral's
                                //    effect on memory use during capture should be small.
                                process_events();
                            }

                            size = round_size(size);
                            auto& pool = get_pool(size, stream);
                            const size_t alloc_size = get_allocation_size(size);
                            AllocParams params(device, size, stream, &pool, alloc_size, stats);
                            params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
                            params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

                            bool block_found =
                                // Search pool
                                get_free_block(params)
                                // Trigger callbacks and retry search
                                || (trigger_free_memory_callbacks(params) && get_free_block(params))
                                // Attempt allocate
                                || alloc_block(params, false)
                                // Free all non-split cached blocks and retry alloc.
                                || (free_cached_blocks() && alloc_block(params, true));

                            if (!block_found) {
                                // For any error code other than cudaErrorMemoryAllocation,
                                // alloc_block should have thrown an exception already.
                                TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

                                size_t device_free;
                                size_t device_total;
                                C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
                                string allowed_info;

                                if (set_fraction) {
                                    allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
                                }

                                stats.num_ooms += 1;

                                // "total capacity": total global memory on GPU
                                // "allowed": memory is allowed to use, which set by fraction.
                                // "already allocated": memory allocated by the program using the
                                //                      caching allocator
                                // "free": free memory as reported by the Cuda API
                                // "cached": memory held by the allocator but not used by the program
                                //
                                // The "allocated" amount  does not include memory allocated outside
                                // of the caching allocator, such as memory allocated by other programs
                                // or memory held by the driver.
                                //
                                // The sum of "allocated" + "free" + "cached" may be less than the
                                // total capacity due to memory held by the driver and usage by other
                                // programs.
                                //
                                // Note that at this point free_cached_blocks has already returned all
                                // possible "cached" memory to the driver. The only remaining "cached"
                                // memory is split from a larger block that is partially in-use.
                                TORCH_CHECK_WITH(
                                    CUDAOutOfMemoryError,
                                    false,
                                    "Cuda out of memory. Tried to allocate ",
                                    format_size(alloc_size),
                                    " (GPU ",
                                    device,
                                    "; ",
                                    format_size(device_total),
                                    " total capacity; ",
                                    format_size(
                                        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                                        .current),
                                        " already allocated; ",
                                        format_size(device_free),
                                        " free; ",
                                        allowed_info,
                                        format_size(
                                            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                                            .current),
                                            " reserved in total by PyTorch)");
                            }

                            TORCH_INTERNAL_ASSERT(
                                params.err == cudaSuccess && params.block != nullptr &&
                                params.block->ptr != nullptr);
                            Block* block = params.block;
                            Block* remaining = nullptr;

                            const bool already_split = block->is_split();
                            if (should_split(block, size)) {
                                remaining = block;

                                block = new Block(device, stream, size, &pool, block->ptr);
                                block->prev = remaining->prev;
                                if (block->prev) {
                                    block->prev->next = block;
                                }
                                block->next = remaining;

                                remaining->prev = block;
                                remaining->ptr = static_cast<char*>(remaining->ptr) + size;
                                remaining->size -= size;
                                bool inserted = pool.blocks.insert(remaining).second;
                                TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

                                if (already_split) {
                                    // An already-split inactive block is being shrunk by size bytes.
                                    update_stat_array(
                                        stats.inactive_split_bytes, -block->size, params.stat_types);
                                } else {
                                    // A new split inactive block is being created from a previously unsplit
                                    // block, size remaining->size bytes.
                                    update_stat_array(
                                        stats.inactive_split_bytes, remaining->size, params.stat_types);
                                    update_stat_array(stats.inactive_split, 1, params.stat_types);
                                }
                            } else if (already_split) {
                                // An already-split block is becoming active
                                update_stat_array(
                                    stats.inactive_split_bytes, -block->size, params.stat_types);
                                update_stat_array(stats.inactive_split, -1, params.stat_types);
                            }

                            block->allocated = true;
                            bool inserted = active_blocks.insert(block).second;
                            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

                            reportMemoryUsageToProfiler(
                                block, block->size, Device(DeviceType::CUDA, device));

                            update_stat_array(stats.allocation, 1, params.stat_types);
                            update_stat_array(stats.allocated_bytes, block->size, params.stat_types);
                            update_stat_array(stats.active, 1, params.stat_types);
                            update_stat_array(stats.active_bytes, block->size, params.stat_types);

                            return block;
                */
            }
            
            pub fn free(&mut self, block: *mut Block)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);

                            block->allocated = false;

                            reportMemoryUsageToProfiler(
                                block, -block->size, Device(DeviceType::CUDA, block->device));

                            StatTypes stat_types;
                            stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
                            stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
                                true;
                            update_stat_array(stats.allocation, -1, {stat_types});
                            update_stat_array(stats.allocated_bytes, -block->size, {stat_types});

                            if (!block->stream_uses.empty()) {
                                if (C10_UNLIKELY(captures_underway)) {
                                    // It's forbidden to cudaEventQuery an event recorded during Cuda graph
                                    // capture. We conservatively defer recording end-of-life events until
                                    // the next call to process_events() (which won't happen until no
                                    // captures are underway)
                                    needs_events_deferred_until_no_capture.push_back(block);
                                } else {
                                    insert_events(block);
                                }
                            } else {
                                free_block(block);
                            }
                */
            }
            
            pub fn get_base_allocation(&mut self, 
                block:    *mut Block,
                out_size: *mut usize)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            while (block->prev) {
                                block = block->prev;
                            }
                            void* basePtr = block->ptr;
                            if (outSize) {
                                size_t size = 0;
                                while (block) {
                                    size += block->size;
                                    block = block->next;
                                }
                                *outSize = size;
                            }
                            return basePtr;
                */
            }
            
            pub fn record_stream(&mut self, 
                block:  *mut Block,
                stream: CudaStream)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            if (stream.stream() == block->stream) {
                                // ignore uses on the allocation stream, since those don't require any
                                // special synchronization
                                return;
                            }
                            block->stream_uses.insert(stream);
                */
            }

            /**
              | set memory fraction to limit maximum
              | allocated memory *
              |
              */
            pub fn set_memory_fraction(&mut self, fraction: f64)  {
                
                todo!();
                /*
                    size_t device_free;
                            size_t device_total;
                            C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
                            allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
                            set_fraction = true;
                */
            }

            /**
              | returns cached blocks to the system
              | allocator *
              |
              */
            pub fn empty_cache(&mut self)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            free_cached_blocks();
                */
            }

            /**
              | Retrieves info (total size + largest
              | block) of the memory cache *
              |
              */
            pub fn cache_info(&mut self, 
                total:   *mut usize,
                largest: *mut usize)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            if (*largest ==
                                0) { // make an initial guess if a zero *largest is passed in
                                size_t tmp_bytes;
                                cudaMemGetInfo(
                                    largest, // Use free memory as an optimistic initial guess of *largest
                                    &tmp_bytes);
                            }
                            cache_info_aux(large_blocks, total, largest);
                            cache_info_aux(small_blocks, total, largest);
                            for (const auto& gp : graph_pools) {
                                cache_info_aux(gp.second->large_blocks, total, largest);
                                cache_info_aux(gp.second->small_blocks, total, largest);
                            }
                */
            }

            /**
              | Returns a copy of the memory allocator
              | stats *
              |
              */
            pub fn get_stats(&mut self) -> DeviceStats {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            return stats;
                */
            }

            /**
              | Resets the historical accumulation
              | stats for the device *
              |
              */
            pub fn reset_accumulated_stats(&mut self)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);

                            for (const auto statType :
                                irange(static_cast<size_t>(StatType::NUM_TYPES))) {
                                reset_accumulated_stat(stats.allocation[statType]);
                                reset_accumulated_stat(stats.segment[statType]);
                                reset_accumulated_stat(stats.active[statType]);
                                reset_accumulated_stat(stats.inactive_split[statType]);
                                reset_accumulated_stat(stats.allocated_bytes[statType]);
                                reset_accumulated_stat(stats.reserved_bytes[statType]);
                                reset_accumulated_stat(stats.active_bytes[statType]);
                                reset_accumulated_stat(stats.inactive_split_bytes[statType]);
                            }

                            stats.num_alloc_retries = 0;
                            stats.num_ooms = 0;
                */
            }

            /**
              | Resets the historical peak stats for
              | the device *
              |
              */
            pub fn reset_peak_stats(&mut self)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);

                            for (const auto statType :
                                irange(static_cast<size_t>(StatType::NUM_TYPES))) {
                                reset_peak_stat(stats.allocation[statType]);
                                reset_peak_stat(stats.segment[statType]);
                                reset_peak_stat(stats.active[statType]);
                                reset_peak_stat(stats.inactive_split[statType]);
                                reset_peak_stat(stats.allocated_bytes[statType]);
                                reset_peak_stat(stats.reserved_bytes[statType]);
                                reset_peak_stat(stats.active_bytes[statType]);
                                reset_peak_stat(stats.inactive_split_bytes[statType]);
                            }
                */
            }

            /**
              | Dump a complete snapshot of the memory
              | held by the allocator. Potentially
              | VERY expensive.
              |
              */
            pub fn snapshot(&self) -> Vec<SegmentInfo> {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);

                            vector<SegmentInfo> result;
                            const auto all_blocks = get_all_blocks();

                            for (const Block* const head_block : all_blocks) {
                                if (head_block->prev != nullptr) {
                                    continue;
                                }
                                result.emplace_back();
                                SegmentInfo& segment_info = result.back();
                                segment_info.device = head_block->device;
                                segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
                                segment_info.is_large = (!head_block->pool->is_small);

                                const Block* block = head_block;
                                while (block != nullptr) {
                                    segment_info.blocks.emplace_back();
                                    BlockInfo& block_info = segment_info.blocks.back();

                                    block_info.size = block->size;
                                    block_info.allocated = block->allocated;
                                    block_info.active = block->allocated || (block->event_count > 0) ||
                                        !block->stream_uses.empty();

                                    segment_info.total_size += block_info.size;
                                    if (block_info.allocated) {
                                        segment_info.allocated_size += block_info.size;
                                    }
                                    if (block_info.active) {
                                        segment_info.active_size += block_info.size;
                                    }

                                    block = block->next;
                                }
                            }

                            sort(
                                result.begin(),
                                result.end(),
                                [](const SegmentInfo& a, const SegmentInfo& b) {
                                    return a.address < b.address;
                                });

                            return result;
                */
            }
            
            pub fn round_size(size: usize) -> usize {
                
                todo!();
                /*
                    if (size < kMinBlockSize) {
                                return kMinBlockSize;
                            } else {
                                return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
                            }
                */
            }

            // See Note [Interaction with Cuda graph capture]

            /// Called by CUDAGraph::capture_begin
            pub fn notify_capture_begin(&mut self, 
                graph_id:   CaptureId,
                mempool_id: MempoolId)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            captures_underway++;
                            auto it = graph_pools.find(mempool_id);
                            if (it == graph_pools.end()) {
                                // mempool_id does not reference an existing pool. Make a new pool for
                                // this capture.
                                graph_pools.emplace(make_pair(
                                        mempool_id, unique_ptr<PrivatePool>(new PrivatePool)));
                            } else {
                                // mempool_id references an existing pool, which the current capture will
                                // share. Check this pool is live (at least one other capture already
                                // references it).
                                TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
                                it->second->use_count++;
                            }
                            // Maps this graph_id to mempool_id and makes sure this graph_id wasn't
                            // somehow assigned a mempool_id already. Keeps essential effect (insert)
                            // out of macro.
                            bool inserted = capture_to_pool_map.insert({graph_id, mempool_id}).second;
                            TORCH_INTERNAL_ASSERT(inserted);
                */
            }

            /// Called by CUDAGraph::capture_end
            ///
            pub fn notify_capture_end(&mut self, graph_id: CaptureId)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            captures_underway--;
                            auto it = capture_to_pool_map.find(graph_id);
                            TORCH_INTERNAL_ASSERT(it != capture_to_pool_map.end());
                            capture_to_pool_map.erase(it);
                */
            }

            /// Called by CUDAGraph::reset
            pub fn notify_capture_destroy(&mut self, mempool_id: MempoolId)  {
                
                todo!();
                /*
                    lock_guard<recursive_mutex> lock(mutex);
                            // The instantiated cudaGraphExec_t has been destroyed. We can't blindly
                            // delete and cudaFree the mempool its capture used, because
                            //  1. other graph(s) might share the same pool
                            //  2. the user might still hold references to output tensors allocated
                            //  during capture.
                            // To handle 1 and 2, we track the number of graphs using this particular
                            // mempool. When the count reaches 0, we tell free_cached_blocks it may now
                            // cudaFree blocks from this graph's pool when it discovers they're unused
                            // (unsplit).
                            auto it = graph_pools.find(mempool_id);
                            TORCH_INTERNAL_ASSERT(it != graph_pools.end());
                            auto uc = --(it->second->use_count);
                            TORCH_INTERNAL_ASSERT(uc >= 0);
                            if (uc == 0) {
                                // Allows free_cached_blocks to begin cudaFreeing this pool's memory,
                                // and makes sure this pool wasn't somehow made freeable already.
                                bool inserted =
                                    graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
                                TORCH_INTERNAL_ASSERT(inserted);
                            }
                */
            }

            /// All private methods do not acquire the
            /// allocator mutex.
            ///
            pub fn get_all_blocks(&self) -> Vec<*const Block> {
                
                todo!();
                /*
                    vector<const Block*> blocks;
                            blocks.insert(
                                blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
                            blocks.insert(
                                blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
                            for (const auto& gp : graph_pools) {
                                blocks.insert(
                                    blocks.end(),
                                    gp.second->small_blocks.blocks.begin(),
                                    gp.second->small_blocks.blocks.end());
                                blocks.insert(
                                    blocks.end(),
                                    gp.second->large_blocks.blocks.begin(),
                                    gp.second->large_blocks.blocks.end());
                            }
                            blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
                            return blocks;
                */
            }

            /**
              | moves a block into a pool of cached free
              | blocks
              |
              */
            pub fn free_block(&mut self, block: *mut Block)  {
                
                todo!();
                /*
                    TORCH_INTERNAL_ASSERT(
                                !block->allocated && block->event_count == 0 &&
                                block->stream_uses.empty());

                            size_t original_block_size = block->size;

                            auto& pool = *block->pool;
                            int64_t net_change_inactive_split_blocks = 0;
                            int64_t net_change_inactive_split_size = 0;

                            const array<Block*, 2> merge_candidates = {block->prev, block->next};
                            for (Block* merge_candidate : merge_candidates) {
                                const int64_t subsumed_size =
                                    try_merge_blocks(block, merge_candidate, pool);
                                if (subsumed_size > 0) {
                                    net_change_inactive_split_blocks -= 1;
                                    net_change_inactive_split_size -= subsumed_size;
                                }
                            }

                            active_blocks.erase(block);
                            // Makes sure the Block* isn't already present in the pool we're freeing it
                            // back into.
                            bool inserted = pool.blocks.insert(block).second;
                            TORCH_INTERNAL_ASSERT(inserted);

                            if (block->is_split()) {
                                net_change_inactive_split_blocks += 1;
                                net_change_inactive_split_size += block->size;
                            }

                            StatTypes stat_types;
                            stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
                            stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
                            update_stat_array(
                                stats.inactive_split, net_change_inactive_split_blocks, stat_types);
                            update_stat_array(
                                stats.inactive_split_bytes, net_change_inactive_split_size, stat_types);
                            update_stat_array(stats.active, -1, stat_types);
                            update_stat_array(stats.active_bytes, -original_block_size, stat_types);
                */
            }

            /**
              | combine previously split blocks. returns
              | the size of the subsumed block, or 0 on
              | failure.
              |
              */
            pub fn try_merge_blocks(&mut self, 
                dst:  *mut Block,
                src:  *mut Block,
                pool: &mut BlockPool) -> usize {
                
                todo!();
                /*
                    if (!src || src->allocated || src->event_count > 0 ||
                                !src->stream_uses.empty()) {
                                return 0;
                            }

                            AT_ASSERT(dst->is_split() && src->is_split());

                            if (dst->prev == src) {
                                dst->ptr = src->ptr;
                                dst->prev = src->prev;
                                if (dst->prev) {
                                    dst->prev->next = dst;
                                }
                            } else {
                                dst->next = src->next;
                                if (dst->next) {
                                    dst->next->prev = dst;
                                }
                            }

                            const size_t subsumed_size = src->size;
                            dst->size += subsumed_size;
                            auto erased = pool.blocks.erase(src);
                            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
                            delete src;

                            return subsumed_size;
                */
            }
            
            pub fn get_pool(&mut self, 
                size:   usize,
                stream: CudaStream) -> &mut BlockPool {
                
                todo!();
                /*
                    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
                            // captures_underway is a conservative guess that the current stream may be
                            // capturing. It's only > 0 if some thread has begun and not yet ended a
                            // capture, so it's usually 0, and we can short-circuit
                            // cudaStreamCaptureStatus (which does a TLS lookup).
                            if (C10_UNLIKELY(captures_underway)) {
                                CaptureId_t id;
                                cudaStreamCaptureStatus status;
                                C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id));
                                if (status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
                                    TORCH_INTERNAL_ASSERT(
                                        status !=
                                        cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated);
                                    // Retrieves the private pool assigned to this capture.
                                    auto it0 = capture_to_pool_map.find(id);
                                    TORCH_INTERNAL_ASSERT(it0 != capture_to_pool_map.end());
                                    auto it1 = graph_pools.find(it0->second);
                                    TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
                                    if (size <= kSmallSize) {
                                        return it1->second->small_blocks;
                                    } else {
                                        return it1->second->large_blocks;
                                    }
                                }
                            }
            #endif
                            if (size <= kSmallSize) {
                                return small_blocks;
                            } else {
                                return large_blocks;
                            }
                */
            }
            
            pub fn get_stat_type_for_pool(&mut self, pool: &BlockPool) -> StatType {
                
                todo!();
                /*
                    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
                */
            }
            
            pub fn should_split(&mut self, 
                block: *const Block,
                size:  usize) -> bool {
                
                todo!();
                /*
                    size_t remaining = block->size - size;
                            return (block->pool->is_small) ? (remaining >= kMinBlockSize)
                                : (remaining > kSmallSize);
                */
            }
            
            pub fn get_allocation_size(size: usize) -> usize {
                
                todo!();
                /*
                    if (size <= kSmallSize) {
                                return kSmallBuffer;
                            } else if (size < kMinLargeAlloc) {
                                return kLargeBuffer;
                            } else {
                                return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
                            }
                */
            }
            
            pub fn get_free_block(&mut self, p: &mut AllocParams) -> bool {
                
                todo!();
                /*
                    BlockPool& pool = *p.pool;
                            auto it = pool.blocks.lower_bound(&p.search_key);
                            if (it == pool.blocks.end() || (*it)->stream != p.stream())
                                return false;
                            p.block = *it;
                            pool.blocks.erase(it);
                            return true;
                */
            }
            
            pub fn trigger_free_memory_callbacks(&mut self, p: &mut AllocParams) -> bool {
                
                todo!();
                /*
                    bool freed_memory = false;
                            for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
                                freed_memory |=
                                    FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
                            }
                            return freed_memory;
                */
            }
            
            pub fn alloc_block(&mut self, 
                p:        &mut AllocParams,
                is_retry: bool) -> bool {
                
                todo!();
                /*
                    // Defensively checks for preexisting Cuda error state.
                            C10_CUDA_CHECK(cudaGetLastError());

                            size_t size = p.alloc_size;
                            void* ptr;

                            if (isRetry) {
                                stats.num_alloc_retries += 1;
                            }

                            if (set_fraction &&
                                total_allocated_memory + size > allowed_memory_maximum) {
                                p.err = cudaErrorMemoryAllocation;
                                return false;
                            } else {
                                p.err = cudaMallocMaybeCapturing(&ptr, size);
                                if (p.err != cudaSuccess) {
                                    if (p.err == cudaErrorMemoryAllocation) {
                                        // If this is the first attempt (!isRetry), we can forgive and clear
                                        // Cuda's
                                        //   internal error state.
                                        // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
                                        // will take
                                        //   over to throw a helpful exception. The user can choose to catch
                                        //   the exception, free some stuff in their script, and attempt their
                                        //   allocation again. In this case, we can also forgive and clear
                                        //   Cuda's internal error state.
                                        cudaGetLastError();
                                    } else {
                                        // If the error's unrelated to memory allocation, we should throw
                                        // immediately.
                                        C10_CUDA_CHECK(p.err);
                                    }
                                    return false;
                                }
                            }

                            if (p.pool->owner_PrivatePool) {
                                // The block is for a Cuda graph's PrivatePool.
                                p.pool->owner_PrivatePool->cudaMalloc_count++;
                            }

                            total_allocated_memory += size;
                            p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
                            update_stat_array(stats.segment, 1, p.stat_types);
                            update_stat_array(stats.reserved_bytes, size, p.stat_types);

                            // p.block came from new, not cudaMalloc. It should not be nullptr here.
                            TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
                            return true;
                */
            }
            
            pub fn free_cached_blocks(&mut self) -> bool {
                
                todo!();
                /*
                    // First ensure that all blocks that can't currently be allocated due to
                            // outstanding events are returned to the pool.
                            synchronize_and_free_events();

                            // Free all non-split cached blocks
                            free_blocks(large_blocks);
                            free_blocks(small_blocks);

                            for (auto it = graph_pools_freeable.begin();
                                it != graph_pools_freeable.end();) {
                                // See notifyCaptureDestroy for the strategy here.
                                TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
                                free_blocks(it->second->small_blocks);
                                free_blocks(it->second->large_blocks);
                                if (it->second->cudaMalloc_count == 0) {
                                    auto erase_count = graph_pools.erase(it->first);
                                    TORCH_INTERNAL_ASSERT(erase_count == 1);
                                    it = graph_pools_freeable.erase(it);
                                } else {
                                    ++it;
                                }
                            }

                            return true;
                */
            }
            
            pub fn free_blocks(&mut self, pool: &mut BlockPool)  {
                
                todo!();
                /*
                    // Frees all non-split blocks
                            auto it = pool.blocks.begin();
                            while (it != pool.blocks.end()) {
                                Block* block = *it;
                                if (!block->prev && !block->next) {
                                    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
                                    total_allocated_memory -= block->size;

                                    if (pool.owner_PrivatePool) {
                                        // The cudaFreed block belonged to a Cuda graph's PrivatePool.
                                        TORCH_INTERNAL_ASSERT(pool.owner_PrivatePool->cudaMalloc_count > 0);
                                        pool.owner_PrivatePool->cudaMalloc_count--;
                                    }

                                    StatTypes stat_types;
                                    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
                                    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
                                    update_stat_array(stats.segment, -1, stat_types);
                                    update_stat_array(stats.reserved_bytes, -block->size, stat_types);

                                    auto cur = it;
                                    ++it;
                                    pool.blocks.erase(cur);
                                    delete block;
                                } else {
                                    ++it;
                                }
                            }
                */
            }
            
            pub fn create_event_internal(&mut self) -> cuda::Event {
                
                todo!();
                /*
                    cudaEvent_t event;
                            C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                            return event;
                */
            }
            
            pub fn free_event_internal(&mut self, event: cuda::Event)  {
                
                todo!();
                /*
                    C10_CUDA_CHECK(cudaEventDestroy(event));
                */
            }
            
            pub fn synchronize_and_free_events(&mut self)  {
                
                todo!();
                /*
                    // Synchronize on outstanding events and then free associated blocks.

                            // This function syncs, so capture should not be underway. Might as well
                            // make sure capture-deferred end of life events get processed too.
                            TORCH_INTERNAL_ASSERT(captures_underway == 0);
                            insert_events_deferred_until_no_capture();

                            for (auto& e : cuda_events) {
                                cudaEvent_t event = e.first;
                                Block* block = e.second;

                                C10_CUDA_CHECK(cudaEventSynchronize(event));
                                free_event_internal(event);

                                block->event_count--;
                                if (block->event_count == 0) {
                                    free_block(block);
                                }
                            }

                            cuda_events.clear();
                */
            }
            
            pub fn insert_events(&mut self, block: *mut Block)  {
                
                todo!();
                /*
                    int prev_device;
                            C10_CUDA_CHECK(cudaGetDevice(&prev_device));

                            stream_set streams(move(block->stream_uses));
                            AT_ASSERT(block->stream_uses.empty());
                            for (auto& stream : streams) {
                                C10_CUDA_CHECK(cudaSetDevice(stream.device_index()));

                                cudaEvent_t event = create_event_internal();
                                C10_CUDA_CHECK(cudaEventRecord(event, stream.stream()));

                                block->event_count++;
                                cuda_events.emplace_back(event, block);
                            }

                            C10_CUDA_CHECK(cudaSetDevice(prev_device));
                */
            }
            
            pub fn insert_events_deferred_until_no_capture(&mut self)  {
                
                todo!();
                /*
                    if (C10_UNLIKELY(needs_events_deferred_until_no_capture.size() > 0)) {
                                for (auto* block : needs_events_deferred_until_no_capture) {
                                    TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
                                    insert_events(block);
                                }
                                needs_events_deferred_until_no_capture.clear();
                            }
                */
            }
            
            pub fn process_events(&mut self)  {
                
                todo!();
                /*
                    insert_events_deferred_until_no_capture();

                            // Process outstanding cudaEvents. Events that are completed are removed
                            // from the queue, and the 'event_count' for the corresponding allocation
                            // is decremented. Stops at the first event which has not been completed.
                            // Since events on different devices or streams may occur out of order,
                            // the processing of some events may be delayed.
                            while (!cuda_events.empty()) {
                                auto& e = cuda_events.front();
                                cudaEvent_t event = e.first;
                                Block* block = e.second;

                                cudaError_t err = cudaEventQuery(event);
                                if (err == cudaErrorNotReady) {
                                    // ignore and clear the error if not ready
                                    cudaGetLastError();
                                    break;
                                } else if (err != cudaSuccess) {
                                    C10_CUDA_CHECK(err);
                                }

                                free_event_internal(event);

                                block->event_count--;
                                if (block->event_count == 0) {
                                    free_block(block);
                                }
                                cuda_events.pop_front();
                            }
                */
            }

            /**
              | Accumulates sizes of all memory blocks
              | for given device in given pool
              |
              */
            pub fn cache_info_aux(&mut self, 
                pool:    &BlockPool,
                total:   *mut usize,
                largest: *mut usize)  {
                
                todo!();
                /*
                    for (const auto& block : pool.blocks) {
                                const auto blocksize = block->size;
                                *total += blocksize;
                                if (blocksize > *largest) {
                                    *largest = blocksize;
                                }
                            }
                */
            }
        }

        pub struct THCCachingAllocator {

            mutex:            RawMutex,

            /**
              | allocated blocks by device pointer
              |
              */
            allocated_blocks: HashMap<*mut c_void,*mut Block>,


            /**
              | lock around calls to cudaFree (to prevent
              | deadlocks with NCCL)
              |
              */
            cuda_free_mutex:  RefCell<RawMutex>,

            device_allocator: Vec<Box<DeviceCachingAllocator>>,
        }

        impl THCCachingAllocator {
            
            pub fn add_allocated_block(&mut self, block: *mut Block)  {
                
                todo!();
                /*
                    lock_guard<mutex> lock(mutex);
                            allocated_blocks[block->ptr] = block;
                */
            }
            
            pub fn get_cuda_free_mutex(&self) -> *mut Mutex {
                
                todo!();
                /*
                    return &cuda_free_mutex;
                */
            }
            
            pub fn get_allocated_block(&mut self, 
                ptr:    *mut c_void,
                remove: bool) -> *mut Block {
                let remove: bool = remove.unwrap_or(false);

                todo!();
                /*
                    lock_guard<mutex> lock(mutex);
                            auto it = allocated_blocks.find(ptr);
                            if (it == allocated_blocks.end()) {
                                return nullptr;
                            }
                            Block* block = it->second;
                            if (remove) {
                                allocated_blocks.erase(it);
                            }
                            return block;
                */
            }
            
            pub fn init(&mut self, device_count: i32)  {
                
                todo!();
                /*
                    const auto size = static_cast<int64_t>(device_allocator.size());
                            if (size < device_count) {
                                device_allocator.resize(device_count);
                                for (const auto i : irange(size, device_count)) {
                                    device_allocator[i] = unique_ptr<DeviceCachingAllocator>(
                                        new DeviceCachingAllocator());
                                }
                            }
                */
            }

            /**
              | allocates a block which is safe to use
              | from the provided stream
              |
              */
            pub fn malloc(&mut self, 
                dev_ptr: *mut *mut c_void,
                device:  i32,
                size:    usize,
                stream:  CudaStream)  {
                
                todo!();
                /*
                    TORCH_INTERNAL_ASSERT(
                                0 <= device && static_cast<size_t>(device) < device_allocator.size(),
                                "Allocator not initialized for device ",
                                device,
                                ": did you call init?");
                            Block* block = device_allocator[device]->malloc(device, size, stream);
                            add_allocated_block(block);
                            *devPtr = (void*)block->ptr;
                */
            }
            
            pub fn free(&mut self, ptr: *mut c_void)  {
                
                todo!();
                /*
                    if (!ptr) {
                                return;
                            }
                            Block* block = get_allocated_block(ptr, true /* remove */);
                            if (!block) {
                                TORCH_CHECK(false, "invalid device pointer: ", ptr);
                            }
                            device_allocator[block->device]->free(block);
                */
            }
            
            pub fn set_memory_fraction(&mut self, 
                fraction: f64,
                device:   i32)  {
                
                todo!();
                /*
                    TORCH_INTERNAL_ASSERT(
                                0 <= device && static_cast<size_t>(device) < device_allocator.size(),
                                "Allocator not initialized for device ",
                                device,
                                ": did you call init?");
                            TORCH_INTERNAL_ASSERT(
                                0 <= fraction && fraction <= 1,
                                "invalid fraction:",
                                fraction,
                                ". Please set within (0, 1).");
                            int activated_device;
                            cudaGetDevice(&activated_device);
                            if (activated_device != device) {
                                cudaSetDevice(device);
                            }
                            device_allocator[device]->setMemoryFraction(fraction);
                */
            }
            
            pub fn empty_cache(&mut self)  {
                
                todo!();
                /*
                    for (auto& da : device_allocator)
                                da->emptyCache();
                */
            }
            
            pub fn get_base_allocation(&mut self, 
                ptr:      *mut c_void,
                out_size: *mut usize)  {
                
                todo!();
                /*
                    Block* block = get_allocated_block(ptr);
                            if (!block) {
                                TORCH_CHECK(false, "invalid device pointer: ", ptr);
                            }
                            return device_allocator[block->device]->getBaseAllocation(block, outSize);
                */
            }
            
            pub fn record_stream(&mut self, 
                ptr:    &DataPtr,
                stream: CudaStream)  {
                
                todo!();
                /*
                    // Empty tensor's storage().data() might be a null ptr. As there is no
                            // blocks associated with those tensors, it is fine to do nothing here.
                            if (!ptr.get()) {
                                return;
                            }

                            // If a tensor is not allocated by this instance, simply skip
                            // This usually happens when Cuda tensors are shared across processes,
                            // we have implemented reference counting based sharing mechanism to
                            // guarantee tensors won't be accidentally freed by one process while
                            // they are still being used in another
                            if (ptr.get_deleter() != &raw_delete)
                                return;

                            Block* block = get_allocated_block(ptr.get());
                            // block must not be null reaching here
                            TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
                            device_allocator[block->device]->recordStream(block, stream);
                */
            }
            
            pub fn snapshot(&mut self) -> Vec<SegmentInfo> {
                
                todo!();
                /*
                    vector<SegmentInfo> result;
                            for (auto& da : device_allocator) {
                                auto snap = da->snapshot();
                                result.insert(result.end(), snap.begin(), snap.end());
                            }

                            return result;
                */
            }
        }

        lazy_static!{
            /*
            THCCachingAllocator caching_allocator;
            */
        }

        /**
          | Returns whether to force all
          | allocations to bypass the caching
          | allocator and go straight to
          | cudaMalloc.
          |
          | This setting is useful when debugging
          | GPU memory errors, since the caching
          | allocator foils cuda-memcheck.
          |
          */
        pub fn force_uncached_allocator() -> bool {
            
            todo!();
                /*
                    static bool force_uncached =
                            getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
                        return force_uncached;
                */
        }

        pub fn uncached_delete(ptr: *mut c_void)  {
            
            todo!();
                /*
                    C10_CUDA_CHECK(cudaFree(ptr));
                */
        }

        /**
          | NB: I decided not to fold this into
          | THCCachingAllocator, because the latter
          | has a lot more methods and it wasn't
          | altogether clear that they should
          | actually be publicly exposed
          |
          */
        pub struct CudaCachingAllocator {
            base: Allocator,
        }

        impl CudaCachingAllocator {
            
            pub fn allocate(&self, size: usize) -> DataPtr {
                
                todo!();
                /*
                    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
                            TORCH_CHECK_WITH(
                                CUDAOutOfMemoryError,
                                size < one_exa_bytes,
                                "Cuda out of memory. Tried to allocate more than 1EB memory.");
                            int device;
                            C10_CUDA_CHECK(cudaGetDevice(&device));
                            void* r = nullptr;
                            if (forceUncachedAllocator()) {
                                // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
                                // if someone tries to use forceUncachedAllocator while capturing.
                                C10_CUDA_CHECK(cudaMalloc(&r, size));
                                return {r, r, &uncached_delete, Device(DeviceType::CUDA, device)};
                            }
                            if (size != 0) {
                                caching_allocator.malloc(
                                    &r, device, size, getCurrentCudaStream(device));
                            }
                            return {r, r, &raw_delete, Device(DeviceType::CUDA, device)};
                */
            }
            
            pub fn raw_deleter(&self) -> DeleterFnPtr {
                
                todo!();
                /*
                    if (forceUncachedAllocator()) {
                                return &uncached_delete;
                            } else {
                                return &raw_delete;
                            }
                */
            }
        }

        lazy_static!{
            /*
            CudaCachingAllocator device_allocator;
            */
        }

        pub fn get() -> *mut Allocator {
            
            todo!();
                /*
                    return &device_allocator;
                */
        }

        pub fn init(device_count: i32)  {
            
            todo!();
                /*
                    caching_allocator.init(device_count);
                */
        }

        pub fn set_memory_fraction(
                fraction: f64,
                device:   i32)  {
            
            todo!();
                /*
                    caching_allocator.setMemoryFraction(fraction, device);
                */
        }

        pub fn empty_cache()  {
            
            todo!();
                /*
                    caching_allocator.emptyCache();
                */
        }

        pub fn cache_info(
                dev_id:          i32,
                cached_and_free: *mut usize,
                largest_block:   *mut usize)  {
            
            todo!();
                /*
                    caching_allocator.device_allocator[dev_id]->cacheInfo(
                            cachedAndFree, largestBlock);
                */
        }

        pub fn get_base_allocation(
                ptr:  *mut c_void,
                size: *mut usize)  {
            
            todo!();
                /*
                    return caching_allocator.getBaseAllocation(ptr, size);
                */
        }

        pub fn record_stream(
                ptr:    &DataPtr,
                stream: CudaStream)  {
            
            todo!();
                /*
                    caching_allocator.recordStream(ptr, stream);
                */
        }

        pub fn get_free_mutex() -> *mut Mutex {
            
            todo!();
                /*
                    return caching_allocator.getCudaFreeMutex();
                */
        }

        #[inline] pub fn assert_valid_device(device: i32)  {
            
            todo!();
                /*
                    const auto device_num = caching_allocator.device_allocator.size();
                        TORCH_CHECK(
                            0 <= device && device < static_cast<int64_t>(device_num),
                            "Invalid device argument.");
                */
        }

        pub fn get_device_stats(device: i32) -> DeviceStats {
            
            todo!();
                /*
                    assertValidDevice(device);
                        return caching_allocator.device_allocator[device]->getStats();
                */
        }

        pub fn reset_accumulated_stats(device: i32)  {
            
            todo!();
                /*
                    assertValidDevice(device);
                        caching_allocator.device_allocator[device]->resetAccumulatedStats();
                */
        }

        pub fn reset_peak_stats(device: i32)  {
            
            todo!();
                /*
                    assertValidDevice(device);
                        caching_allocator.device_allocator[device]->resetPeakStats();
                */
        }

        pub fn snapshot() -> Vec<SegmentInfo> {
            
            todo!();
                /*
                    return caching_allocator.snapshot();
                */
        }

        /**
          | CUDAGraph interactions
          |
          */
        pub fn notify_capture_begin(
                device:     i32,
                graph_id:   CaptureId,
                mempool_id: MempoolId)  {
            
            todo!();
                /*
                    assertValidDevice(device);
                        caching_allocator.device_allocator[device]->notifyCaptureBegin(
                            graph_id, mempool_id);
                */
        }

        pub fn notify_capture_end(
                device:   i32,
                graph_id: CaptureId)  {
            
            todo!();
                /*
                    assertValidDevice(device);
                        caching_allocator.device_allocator[device]->notifyCaptureEnd(graph_id);
                */
        }

        pub fn notify_capture_destroy(
                device:     i32,
                mempool_id: MempoolId)  {
            
            todo!();
                /*
                    assertValidDevice(device);
                        caching_allocator.device_allocator[device]->notifyCaptureDestroy(mempool_id);
                */
        }

        /**
          | In Cuda IPC, sender sends a tensor to
          | receiver, getIpcDevPtr is called by the
          | receiving process to map the Cuda
          | memory from the sending process into
          | its own address space.
          |
          | Cuda IPC only allows sharing a big
          | memory block associated with
          | a cudaIpcMemHandle_t and it can be
          | opened only **once** per context per
          | process. There can be multiple types of
          | storage in the same IPC mem block, so
          | we must cache the device ptr to
          | construct typed storage as it comes.
          |
          | ipcMemHandle_to_devptr maps
          | a cudaIpcMemHandle_t to a device
          | pointer in the process that can be used
          | to access the memory block in the
          | sender process. It only saves
          | a weak_ptr of the device pointer in the
          | map, the shared_ptr will be used to
          | reconstruct all storages in this
          | CudaMalloc allocation. And it will
          | deleted in cudaIpcCloseMemHandle when
          | its reference count is 0.
          |
          */
        lazy_static!{
            /*
            mutex IpcMutex;
            unordered_map<string, weak_ptr<void>> ipcMemHandle_to_devptr;
            */
        }

        pub fn get_ipc_dev_ptr(handle: String) -> Arc<c_void> {
            
            todo!();
                /*
                    lock_guard<mutex> lock(IpcMutex);

                        auto iter = ipcMemHandle_to_devptr.find(handle);
                        if (iter != ipcMemHandle_to_devptr.end()) {
                            auto devptr = iter->second.lock();
                            if (devptr)
                                return devptr;
                        }
                        // This ipcMemHandle hasn't been opened, or already expired, open it to
                        // enable IPC access to that mem block.
                        void* dev = nullptr;
                        auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
                        C10_CUDA_CHECK(
                            cudaIpcOpenMemHandle(&dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
                        // devPtr has to be deleted in same device when created.
                        int curr_device;
                        C10_CUDA_CHECK(cudaGetDevice(&curr_device));
                        auto sp = shared_ptr<void>(dev, [handle, curr_device](void* ptr) {
                            CUDAGuard device_guard(curr_device);
                            lock_guard<mutex> deleter_lock(IpcMutex);
                            C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
                            ipcMemHandle_to_devptr.erase(handle);
                        });
                        weak_ptr<void> wp = sp;
                        // To eliminate an additional search, we can use insert().
                        // It doesn't overwrite when key already exists(ptr expired).
                        // But in the deleter for sp we erased the entry,
                        // this should be safe to do now.
                        ipcMemHandle_to_devptr.insert(iter, {handle, wp});

                        return sp;
                */
        }

        pub fn raw_alloc(nbytes: usize)  {
            
            todo!();
                /*
                    if (nbytes == 0) {
                            return nullptr;
                        }
                        int device;
                        C10_CUDA_CHECK(cudaGetDevice(&device));
                        void* r = nullptr;
                        caching_allocator.malloc(
                            &r, device, nbytes, getCurrentCudaStream(device));
                        return r;
                */
        }

        pub fn raw_alloc_with_stream(
                nbytes: usize,
                stream: CudaStream)  {
            
            todo!();
                /*
                    if (nbytes == 0) {
                            return nullptr;
                        }
                        int device;
                        C10_CUDA_CHECK(cudaGetDevice(&device));
                        void* r = nullptr;
                        caching_allocator.malloc(&r, device, nbytes, stream);
                        return r;
                */
        }

        pub fn raw_delete(ptr: *mut c_void)  {
            
            todo!();
                /*
                    caching_allocator.free(ptr);
                */
        }
    }

}

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDACachingAllocator.cpp]

c10_define_registry!{
    FreeCudaMemoryCallbacksRegistry, 
    FreeMemoryCallback,
}
