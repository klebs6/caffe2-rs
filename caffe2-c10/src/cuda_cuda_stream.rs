/*!
 | Stream pool note.
 |
 | A CudaStream is an abstraction of an actual
 | cuStream on the GPU. CudaStreams are backed by
 | cuStreams, but they use several pools to
 | minimize the costs associated with creating,
 | retaining, and destroying cuStreams.
 |
 | There are three pools per device, and
 | a device's pools are lazily created.
 |
 | The first pool contains only the default
 | stream. When the default stream is requested
 | it's returned.
 |
 | The second pool is the "low priority" or
 | "default priority" streams. In HIP builds there
 | is no distinction between streams in this pool
 | and streams in the third pool (below). There
 | are 32 of these streams per device, and when
 | a stream is requested one of these streams is
 | returned round-robin. That is, the first stream
 | requested is at index 0, the second at index
 | 1... to index 31, then index 0 again.
 |
 | This means that if 33 low priority streams are
 | requested, the first and last streams requested
 | are actually the same stream (under the covers)
 | and kernels enqueued on them cannot run
 | concurrently.
 |
 | The third pool is the "high priority"
 | streams. The third pool acts like the second
 | pool except the streams are created with
 | a higher priority.
 |
 | These pools suggest that stream users should
 | prefer many short-lived streams, as the cost of
 | acquiring and releasing streams is effectively
 | zero. If many longer-lived streams are required
 | in performance critical scenarios then the
 | functionality here may need to be extended to
 | allow, for example, "reserving" a subset of the
 | pool so that other streams do not accidentally
 | overlap the performance critical streams.
 |
 | Note: although the notion of "current stream
 | for device" is thread local (every OS thread
 | has a separate current stream, as one might
 | expect), the stream pool is global across all
 | threads; stream 0 is always stream 0 no matter
 | which thread you use it on.  Multiple threads
 | can synchronize on the same stream.  Although
 | the Cuda documentation is not very clear on the
 | matter, streams are thread safe; e.g., it is
 | safe to enqueue a kernel on the same stream
 | from two different threads.
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CudaStream.h]

#[cfg(feature = "cuda")]
pub enum CudaStreamUnchecked { 
    UNCHECKED 
}

/**
  | Value object representing a Cuda stream.
  |
  | This is just a wrapper around Stream, but it
  | comes with a little extra Cuda-specific
  | functionality (conversion to cudaStream_t), and
  | a guarantee that the wrapped Stream really is
  | a Cuda stream.
  |
  */
#[cfg(feature = "cuda")]
#[derive(PartialEq,Eq)]
pub struct CudaStream {
    stream: Stream,
}

#[cfg(feature = "cuda")]
impl CudaStream {

    /**
      | Construct a CudaStream from a Stream.  This
      | construction is checked, and will raise an
      | error if the Stream is not, in fact, a Cuda
      | stream.
      |
      */
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : stream(stream),

            TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
        */
    }

    /**
      | Construct a CudaStream from a Stream with no
      | error checking.
      |
      | This constructor uses the "named"
      | constructor idiom, and can be invoked as:
      | CudaStream(CudaStream::UNCHECKED, stream)
      */
    pub fn new(
        _0:     Unchecked,
        stream: Stream) -> Self {
    
        todo!();
        /*
        : stream(stream),

        
        */
    }

    /// Implicit conversion to cudaStream_t.
    pub fn operator_cuda_stream_t(&self) -> CudaStream {
        
        todo!();
        /*
            return stream();
        */
    }

    /**
      | Implicit conversion to Stream (a.k.a.,
      | forget that the stream is a Cuda stream).
      |
      */
    pub fn operator_stream(&self) -> Stream {
        
        todo!();
        /*
            return unwrap();
        */
    }

    /// Get the Cuda device index that this stream
    /// is associated with.
    ///
    pub fn device_index(&self) -> DeviceIndex {
        
        todo!();
        /*
            return stream_.device_index();
        */
    }

    /**
      | Get the full Device that this stream is
      | associated with.  The Device is guaranteed
      | to be a Cuda device.
      |
      */
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return Device(DeviceType::CUDA, device_index());
        */
    }

    /// Return the stream ID corresponding to this
    /// particular stream.
    ///
    pub fn id(&self) -> StreamId {
        
        todo!();
        /*
            return stream_.id();
        */
    }
    
    pub fn query(&self) -> bool {
        
        todo!();
        /*
            DeviceGuard guard{stream_.device()};
        cudaError_t err = cudaStreamQuery(stream());

        if (err == cudaSuccess) {
          return true;
        } else if (err != cudaErrorNotReady) {
          C10_CUDA_CHECK(err);
        }

        return false;
        */
    }
    
    pub fn synchronize(&self)  {
        
        todo!();
        /*
            DeviceGuard guard{stream_.device()};
        C10_CUDA_CHECK(cudaStreamSynchronize(stream()));
        */
    }
    
    pub fn priority(&self) -> i32 {
        
        todo!();
        /*
            DeviceGuard guard{stream_.device()};
        int priority = 0;
        C10_CUDA_CHECK(cudaStreamGetPriority(stream(), &priority));
        return priority;
        */
    }

    /// Explicit conversion to cudaStream_t.
    pub fn stream(&self) -> CudaStream {
        
        todo!();
        /*
        
        */
    }

    /// Explicit conversion to Stream.
    pub fn unwrap(&self) -> Stream {
        
        todo!();
        /*
            return stream_;
        */
    }

    /**
      | Reversibly pack a CudaStream into a uint64_t
      | representation.  This may be helpful when
      | storing a CudaStream in a C struct, where
      | you cannot conveniently place the CudaStream
      | object itself (which is morally equivalent,
      | but unfortunately is not POD due to the fact
      | that it has constructors.)
      |
      | The CudaStream can be unpacked using
      | unpack().  The format of the uint64_t is
      | unspecified and may be changed.
      */
    pub fn pack(&self) -> u64 {
        
        todo!();
        /*
            return stream_.pack();
        */
    }

    /**
      | Unpack a CudaStream from the uint64_t
      | representation generated by pack().
      |
      */
    pub fn unpack(bits: u64) -> CudaStream {
        
        todo!();
        /*
            return CudaStream(Stream::unpack(bits));
        */
    }
    
    pub fn priority_range() -> (i32,i32) {
        
        todo!();
        /*
            // Note: this returns the range of priority **supported by PyTorch**, not
        // the range of priority **supported by Cuda**. The former is a subset of
        // the latter. Currently PyTorch only supports 0 and -1, which are "low" and
        // "high" priority.
        int least_priority, greatest_priority;
        C10_CUDA_CHECK(
            cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        TORCH_INTERNAL_ASSERT(
            least_priority >= 0, "Unexpected Cuda stream priority range");
        TORCH_INTERNAL_ASSERT(
            greatest_priority <= -1, "Unexpected Cuda stream priority range");
        return make_tuple(0, -1);
        */
    }
}

lazy_static!{
    /*
    struct hash<CudaStream> {
      size_t operator()(CudaStream s) const  {
        return hash<Stream>{}(s.unwrap());
      }
    };
    */
}

//-------------------------------------------[.cpp/pytorch/c10/cuda/CudaStream.cpp]

/**
  | Internal implementation that leaks the
  | stream. It's not intended to be used outside of
  | this file.
  |
  */
#[cfg(feature = "cuda")]
struct LeakyStreamInternals {
    device_index: DeviceIndex,  // default = -1
    stream_id:    int32_t,      // default = -1
    stream:       CudaStream, // default = nullptr
}

#[cfg(feature = "cuda")]
impl Drop for LeakyStreamInternals {

    fn drop(&mut self) {
        todo!();
        /*
            // NB: this code is invoked only in the destruction of global variables
        // (since we never shrink the corresponding vectors). At this point the Cuda
        // runtime might be already destroyed and invoking cudaStreamDestroy leads
        // to a crash. It's likely an issue in Cuda, but to be safe - let's just
        // "forget" the destruction.

        // if (stream) cudaStreamDestroy(stream);
        */
    }
}

lazy_static!{
    /*
    // Global stream state and constants
    static DeviceIndex num_gpus = -1;
    static constexpr int kStreamsPerPoolBits = 5;
    static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
    static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;
    static constexpr int kStreamTypeBits = 3;

    // Note: lower numbers are higher priorities, zero is default priority
    static int kHighPriority = -1;
    static int kLowPriority = 0;

    // Default streams
    static once_flag init_flag;
    static LeakyStreamInternals default_streams[C10_COMPILE_TIME_MAX_GPUS];

    // Non-default streams
    // Note: the number of Cuda devices is determined at run time,
    // and the low and high priority pools are lazily initialized
    // when the first stream is requested for a device.
    // The device flags track the initialization of each device, while
    // the low and high priority counters track, for each device, the next stream
    // in the pool to be returned when a stream is requested (round-robin fashion
    // , see the note in CudaStream.h).
    //
    // unique_ptr<T[]> is used instead of vector<T> because T might be non-movable
    // and non-copyable.
    static once_flag device_flags[C10_COMPILE_TIME_MAX_GPUS];
    static atomic<uint32_t> low_priority_counters[C10_COMPILE_TIME_MAX_GPUS];
    static atomic<uint32_t> high_priority_counters[C10_COMPILE_TIME_MAX_GPUS];
    static array<LeakyStreamInternals, kStreamsPerPool> low_priority_streams[C10_COMPILE_TIME_MAX_GPUS];
    static array<LeakyStreamInternals, kStreamsPerPool> high_priority_streams[C10_COMPILE_TIME_MAX_GPUS];
    */
}

/**
  | Note [StreamId assignment]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~
  | How do we assign stream IDs?
  |
  | -- 57 bits --  -- 5 bits -----  -- 3 bits --
  | zeros          stream id index  StreamIdType
  |
  | Where StreamIdType:
  |  000 = default stream or externally allocated if id[63:3] != 0
  |  001 = low priority stream
  |  010 = high priority stream
  |
  | This is not really for efficiency; it's just
  | easier to write the code to extract the index
  | if we do this with bitmasks :)
  |
  | We are obligated to treat the stream ID 0 as
  | the default stream, per the invariant specified
  | in Stream.
  |
  | However, all other numbers are entirely an
  | internal implementation detail, we reserve the
  | right to renumber streams however we like.
  |
  | Note that it is really important that the MSB
  | is zero; StreamId is a *signed* integer, and
  | unsigned to signed conversion outside of the
  | bounds of signed integer representation is
  | undefined behavior.
  |
  | You could work around this with something like
  | https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
  | but it seems a bit overkill for this.
  |
  | Also, external managed stream pointers
  | (cudaStream_t) can be directly stored in the Id
  | field so in this case, we need to check the
  | stream alignment.
  |
  | The IdType uses an additional bit to match with
  | the 64-bit address alignment making easy to
  | identify an external stream when its value (X
  | & 7) > 0
  |
  */
#[repr(u8)]
pub enum StreamIdType {
    DEFAULT = 0x0,
    LOW     = 0x1,
    HIGH    = 0x2,
    EXT     = 0x3,
}

impl fmt::Display for StreamIdType {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            switch (s) {
        case StreamIdType::DEFAULT:
          stream << "DEFAULT";
          break;
        case StreamIdType::LOW:
          stream << "LOW";
          break;
        case StreamIdType::HIGH:
          stream << "HIGH";
          break;
        case StreamIdType::EXT:
          stream << "EXT";
          break;
        default:
          stream << static_cast<uint8_t>(s);
          break;
      }
      return stream;
        */
    }
}

/**
  | StreamId is 64-bit, so we can just rely on
  | regular promotion rules.
  |
  | We rely on streamIdIndex and streamIdType being
  | non-negative; see Note [Hazard when
  | concatenating signed integers]
  */
#[inline] pub fn stream_id_type(s: StreamId) -> StreamIdType {
    
    todo!();
        /*
            int mask_for_type = (1 << kStreamTypeBits) - 1;
      if (s && ((s & mask_for_type) == 0)) {
        // Externally allocated streams have their id being the cudaStream_ptr
        // so the bits corresponding to the type will be 0 and will collide with
        // the default stream.
        return StreamIdType::EXT;
      }
      return static_cast<StreamIdType>(s & mask_for_type);
        */
}

#[inline] pub fn stream_id_index(s: StreamId) -> usize {
    
    todo!();
        /*
            return static_cast<size_t>(
          (s >> kStreamTypeBits) & ((1 << kStreamsPerPoolBits) - 1));
        */
}

pub fn make_stream_id(
        st: StreamIdType,
        si: usize) -> StreamId {
    
    todo!();
        /*
            return (static_cast<StreamId>(si) << kStreamTypeBits) |
          static_cast<StreamId>(st);
        */
}

pub fn pointer_within<T, A>(
    ptr: *const T,
    arr: &A) -> bool {

    todo!();
        /*
            return greater_equal<const T*>()(ptr, arr.data()) &&
          less<const T*>()(ptr, arr.data() + arr.size());
        */
}

#[cfg(feature = "cuda")]
pub fn cuda_stream_get_stream_id(ptr: *const LeakyStreamInternals) -> StreamId {
    
    todo!();
        /*
            // Hypothetically, we could store the stream ID in the stream.  But that
      // introduces a degree of freedom which could lead to bugs (where we
      // misnumber streams in the pool, or overwrite the number).  Better
      // to just compute it based on the metric that actually matters,
      // which is how we map IDs back into the vectors.

      DeviceIndex device_index = ptr->device_index;

      // Check if it's the default stream
      if (ptr == &default_streams[device_index]) {
        return makeStreamId(StreamIdType::DEFAULT, 0);
      }

      // Check if it's a low priority stream
      // NB: Because ptr may not necessarily lie within the array, we must use
      // less and similar templates to avoid UB that arises when
      // doing an operator< comparison.
      if (pointer_within<LeakyStreamInternals>(
              ptr, low_priority_streams[device_index])) {
        return makeStreamId(
            StreamIdType::LOW, ptr - low_priority_streams[device_index].data());
      }

      // Check if it's a high priority stream
      if (pointer_within<LeakyStreamInternals>(
              ptr, high_priority_streams[device_index])) {
        return makeStreamId(
            StreamIdType::HIGH, ptr - high_priority_streams[device_index].data());
      }

      TORCH_INTERNAL_ASSERT(
          0,
          "Could not compute stream ID for ",
          ptr,
          " on device ",
          device_index,
          " (something has gone horribly wrong!)");
        */
}

/// Thread-local current streams
lazy_static!{
    /*
    static thread_local LeakyStreamInternals** current_streams = nullptr;
    */
}

/**
  | Populates global values and creates a default
  | stream for each device.
  |
  | Note: the default stream on each device is
  | signified by a nullptr, and so is not created
  | as usual.
  |
  | In particular, we don't need to switch devices
  | when creating the streams.
  |
  | Warning: this function must only be called
  | once!
  |
  */
pub fn init_global_stream_state()  {
    
    todo!();
        /*
            num_gpus = device_count();
      // Check if the number of GPUs matches the expected compile-time max number
      // of GPUs.
      TORCH_CHECK(
          num_gpus <= C10_COMPILE_TIME_MAX_GPUS,
          "Number of Cuda devices on the machine is larger than the compiled "
          "max number of gpus expected (",
          C10_COMPILE_TIME_MAX_GPUS,
          "). Increase that and recompile.");

      // Initializes default streams
      for (const auto i : irange(num_gpus)) {
        default_streams[i].device_index = i;
        low_priority_counters[i] = 0;
        high_priority_counters[i] = 0;
      }
        */
}

/**
  | Creates the low and high priority stream pools
  | for the specified device
  |
  | Warning: only call once per device!
  |
  */
pub fn init_device_stream_state(device_index: DeviceIndex)  {
    
    todo!();
        /*
            // Switches to the requested device so streams are properly associated
      // with it.
      CUDAGuard device_guard{device_index};

      for (const auto i : irange(kStreamsPerPool)) {
        auto& lowpri_stream = low_priority_streams[device_index][i];
        auto& hipri_stream = high_priority_streams[device_index][i];

        lowpri_stream.device_index = device_index;
        hipri_stream.device_index = device_index;

        C10_CUDA_CHECK(cudaStreamCreateWithPriority(
            &lowpri_stream.stream, kDefaultFlags, kLowPriority));
        C10_CUDA_CHECK(cudaStreamCreateWithPriority(
            &hipri_stream.stream, kDefaultFlags, kHighPriority));
      }
        */
}

/**
  | Init front-end to ensure initialization
  | only occurs once
  |
  */
#[cfg(feature = "cuda")]
pub fn init_cuda_streams_once()  {
    
    todo!();
        /*
            // Inits default streams (once, globally)
      call_once(init_flag, initGlobalStreamState);

      if (current_streams) {
        return;
      }

      // Inits current streams (thread local) to default streams
      current_streams =
          (LeakyStreamInternals**)malloc(num_gpus * sizeof(LeakyStreamInternals*));
      for (const auto i : irange(num_gpus)) {
        current_streams[i] = &default_streams[i];
      }
        */
}

/**
  | Helper to verify the GPU index is valid
  |
  */
#[inline] pub fn check_gpu(device_index: DeviceIndex)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_gpus);
        */
}

/**
  | Helper to determine the index of the stream to
  | return
  |
  | Note: Streams are returned round-robin (see
  | note in CudaStream.h)
  |
  */
pub fn get_idx(counter: &mut Atomic<u32>) -> u32 {
    
    todo!();
        /*
            auto raw_idx = counter++;
      return raw_idx % kStreamsPerPool;
        */
}

/// See Note [StreamId assignment]
///
#[cfg(feature = "cuda")]
pub fn cuda_stream_internals(s: CudaStream) -> *mut LeakyStreamInternals {
    
    todo!();
        /*
            DeviceIndex device_index = s.device_index();
      StreamIdType st = streamIdType(s.unwrap().id());
      size_t si = streamIdIndex(s.unwrap().id());
      switch (st) {
        case StreamIdType::DEFAULT:
          TORCH_INTERNAL_ASSERT(
              si == 0,
              "Unrecognized stream ",
              s.unwrap(),
              " (I think this should be the default stream, but I got a non-zero index ",
              si,
              ").",
              " Did you manufacture the StreamId yourself?  Don't do that; use the",
              " official API like getStreamFromPool() to get a new stream.");
          return &default_streams[device_index];
        case StreamIdType::LOW:
          return &low_priority_streams[device_index][si];
        case StreamIdType::HIGH:
          return &high_priority_streams[device_index][si];
        default:
          TORCH_INTERNAL_ASSERT(
              0,
              "Unrecognized stream ",
              s.unwrap(),
              " (I didn't recognize the stream type, ",
              st,
              ")");
      }
        */
}

#[cfg(feature = "cuda")]
pub fn cuda_stream_from_internals(ptr: *const LeakyStreamInternals) -> CudaStream {
    
    todo!();
        /*
            return CudaStream(
          CudaStream::UNCHECKED,
          Stream(
              Stream::UNSAFE,
              Device(DeviceType::CUDA, ptr->device_index),
              CudaStream_getStreamId(ptr)));
        */
}

#[cfg(feature = "cuda")]
impl CudaStream {
    
    pub fn stream(&self) -> CudaStream {
        
        todo!();
        /*
            int64_t stream_id = unwrap().id();
      if (streamIdType(stream_id) == StreamIdType::EXT) {
        // In this case this is a externally allocated stream
        // we don't need to manage its life cycle
        return reinterpret_cast<cudaStream_t>(stream_id);
      } else {
        auto ptr = CudaStream_internals(*this);
        TORCH_INTERNAL_ASSERT(ptr);
        return ptr->stream;
      }
        */
    }
}

/**
  | Get a new stream from the Cuda stream
  | pool. You can think of this as "creating"
  | a new stream, but no such creation actually
  | happens; instead, streams are preallocated
  | from the pool and returned in a round-robin
  | fashion.
  | 
  | You can request a stream from the high
  | priority pool by setting isHighPriority
  | to true, or a stream for a specific device
  | by setting device (defaulting to the
  | current Cuda stream.)
  |
  | Returns a stream from the requested pool
  |
  | Note: when called the first time on a device,
  | this will create the stream pools for that
  | device.
  |
  */
#[cfg(feature = "cuda")]
pub fn get_stream_from_pool(
        is_high_priority: bool,
        device_index:     DeviceIndex) -> CudaStream {

    let is_high_priority: bool = is_high_priority.unwrap_or(false);
    let device: DeviceIndex = device.unwrap_or(-1);
    
    todo!();
        /*
            initCudaStreamsOnce();
      if (device_index == -1)
        device_index = current_device();
      check_gpu(device_index);

      // Initializes the stream pools (once)
      call_once(
          device_flags[device_index], initDeviceStreamState, device_index);

      if (isHighPriority) {
        const auto idx = get_idx(high_priority_counters[device_index]);
        return CudaStream_fromInternals(&high_priority_streams[device_index][idx]);
      }

      const auto idx = get_idx(low_priority_counters[device_index]);
      return CudaStream_fromInternals(&low_priority_streams[device_index][idx]);
        */
}

/**
  | Get a CudaStream from a externally allocated
  | one.
  | 
  | This is mainly for interoperability
  | with different libraries where we want
  | to operate on a non-torch allocated
  | stream for data exchange or similar
  | purposes
  |
  */
#[cfg(feature = "cuda")]
pub fn get_stream_from_external(
        ext_stream:   CudaStream,
        device_index: DeviceIndex) -> CudaStream {
    
    todo!();
        /*
            return CudaStream(
          CudaStream::UNCHECKED,
          // The stream pointer will be the actual id
          Stream(
              Stream::UNSAFE,
              Device(DeviceType::CUDA, device_index),
              reinterpret_cast<int64_t>(ext_stream)));
        */
}

/**
  | Get the default Cuda stream, for the
  | passed Cuda device, or for the current
  | device if no device index is passed.
  | The default stream is where most computation
  | occurs when you aren't explicitly using
  | streams.
  |
  */
#[cfg(feature = "cuda")]
pub fn get_default_cuda_stream(device_index: DeviceIndex) -> CudaStream {

    let device_index: DeviceIndex = device_index.unwrap_or(-1);
    
    todo!();
        /*
            initCudaStreamsOnce();
      if (device_index == -1) {
        device_index = current_device();
      }
      check_gpu(device_index);
      return CudaStream_fromInternals(&default_streams[device_index]);
        */
}

/**
  | Get the current Cuda stream, for the
  | passed Cuda device, or for the current
  | device if no device index is passed.
  | 
  | The current Cuda stream will usually
  | be the default Cuda stream for the device,
  | but it may be different if someone called
  | 'setCurrentCudaStream' or used 'StreamGuard'
  | or 'CudaStreamGuard'.
  |
  */
#[cfg(feature = "cuda")]
pub fn get_current_cuda_stream(device_index: DeviceIndex) -> CudaStream {

    let device_index: DeviceIndex = device_index.unwrap_or(-1);
    
    todo!();
        /*
            initCudaStreamsOnce();
      if (device_index == -1) {
        device_index = current_device();
      }
      check_gpu(device_index);
      return CudaStream_fromInternals(current_streams[device_index]);
        */
}

/**
  | Set the current stream on the device
  | of the passed in stream to be the passed
  | in stream. Yes, you read that right:
  | this function has *nothing* to do with
  | the current device: it toggles the current
  | stream of the device of the passed stream.
  | 
  | Confused? Avoid using this function;
  | prefer using 'CudaStreamGuard' instead
  | (which will switch both your current
  | device and current stream in the way
  | you expect, and reset it back to its original
  | state afterwards).
  |
  */
#[cfg(feature = "cuda")]
pub fn set_current_cuda_stream(stream: CudaStream)  {
    
    todo!();
        /*
            initCudaStreamsOnce();
      auto ptr = CudaStream_internals(stream);
      TORCH_INTERNAL_ASSERT(ptr);
      current_streams[ptr->device_index] = ptr;
        */
}

#[cfg(feature = "cuda")]
impl fmt::Display for CudaStream {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return stream << s.unwrap();
        */
    }
}
