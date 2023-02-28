crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Tensor.h]

pub struct VulkanTensorFuture<Type,const kAccess: AccessFlags> {
    tensor: *const VulkanTensor,
}

impl<Type,const kAccess: AccessFlags> HasPayload 
for VulkanTensorFuture<Type,kAccess> 
{
    type Payload = MemoryHandle<AccessPointer<Type,kAccess>>;
}

impl<Type,const kAccess: AccessFlags> VulkanTensorFuture<Type,kAccess> {
    
    pub fn new(tensor: *const VulkanTensor) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn new(_0: Future) -> Self {
    
        todo!();
        /*
        
       */
    }
    
    pub fn assign_from(&mut self, _0: Future) -> &mut Future {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(_0: Future<T,A>) -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | This is a blocking operation as the name
      | suggests.  A call to host() will trigger an
      | async copy if pending writes are detected.
      |
      | Consequently, for optimal performance, put
      | as much time and distance between the place
      | where a VulkanTensor::host() call occurs and the
      | location where the returned future is
      | explicitly waited on as a result of a call
      | to this function.
      |
      | // Intentionally disabed to enforce a usage
      | // pattern wherein the Future's lifetime
      | // exceeds that of the Payload as we use the
      | // Future's destructor to eagerly (as opposed
      | // to lazily and upon first use) upload the
      | // modifications back onto the GPU in an
      | // effort to hide the upload latency.
      |
      | Payload wait() const && = delete;
      |
      */
    pub fn wait(&self) -> Payload {
        
        todo!();
        /*
        
        */
    }
}

pub struct VulkanTensorViewStateBundleBuffer {
    stage:  VkPipelineStageFlags,
    access: VkAccessFlags,
}

impl VulkanTensorViewStateBundleBuffer {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

pub struct VulkanTensorViewStateBundleImage {
    stage:  VkPipelineStageFlags,
    access: VkAccessFlags,
    layout: VkImageLayout,
}

impl VulkanTensorViewStateBundleImage {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

pub struct VulkanTensorViewStateBundle {
    staging: VulkanTensorViewStateBundleBuffer,
    buffer:  VulkanTensorViewStateBundleBuffer,
    image:   VulkanTensorViewStateBundleImage,
}

bitflags!{
    pub struct VulkanTensorViewStateComponentType : u8 {
        const Buffer  = 1;
        const Image   = 2;
        const Staging = 4;
        const All     = Self::Buffer | Self::Image | Self::Staging;
    }
}

pub struct VulkanTensorViewStateComponent {

}

impl HasFlags for VulkanTensorViewStateComponent {
    type Flags = VulkanTensorViewStateComponentType;
}

#[derive(Default)]
pub struct VulkanTensorViewState {
    available: ComponentFlags,
    dirty:     ComponentFlags,
    bundle:    Bundle,
}

impl HasTransition for VulkanTensorViewState {
    type Transition = (Bundle,Bundle);
}

impl VulkanTensorViewState {

    pub fn new(
        _0: *const Adapter,
        _1: &[i32]) -> Self {
    
        todo!();
        /*


        
        */
    }

    // Availability
    pub fn is_available(&self, _0: ComponentFlags) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_discrete(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_uma(&self) -> bool {
        
        todo!();
        /*
        
        */
    }

    // Clean / Dirty
    pub fn is_clean(&self, _0: ComponentFlags) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_dirty(&self, _0: ComponentFlags) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_clean(&mut self, _0: ComponentFlags)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_dirty(&mut self, _0: ComponentFlags)  {
        
        todo!();
        /*
        
        */
    }

    pub fn transition(&mut self, to: Bundle) -> Transition {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------
pub struct VulkanTensorView {

    /**
      | Resources
      |
      */
    buffer:  RefCell<Buffer>,
    image:   RefCell<Image>,
    staging: RefCell<Buffer>,
    fence:   RefCell<Fence>,

    /**
      | Context
      |
      */
    context: *mut Context,

    pool:    *mut ResourcePool,

    /**
      | State
      |
      */
    state:   RefCell<State>,

    /**
      | Metadata
      |
      */
    extents: Uvec3,

    options: TensorOptions,
    sizes:   SmallVector<i64,6>,
    strides: SmallVector<i64,6>,
}

impl Default for View {
    
    fn default() -> Self {
        todo!();
        /*


        
        */
    }
}

impl HasComponent for VulkanTensorView {
    type Component = StateComponent;
}

impl VulkanTensorView {
    
    pub fn new(
        context: *mut Context,
        pool:    *mut ResourcePool,
        sizes:   &[i32],
        options: &TensorOptions) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn buffer(&self, 
        _0: &mut CommandBuffer,
        _1: StageFlags,
        _2: AccessFlags) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_image(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&self, 
        _0: &mut CommandBuffer,
        _1: StageFlags,
        _2: AccessFlags) -> &mut Image {
        
        todo!();
        /*
        
        */
    }
    
    pub fn staging(&self, 
        _0: &mut CommandBuffer,
        _1: StageFlags,
        _2: AccessFlags) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn wait(&self) -> &mut VulkanTensorMemory {
        
        todo!();
        /*
        
        */
    }
    
    pub fn extents(&self) -> &Uvec3 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn options(&self) -> &TensorOptions {
        
        todo!();
        /*
        
        */
    }
    
    pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }

    // Accessors / Lazy Allocation
    pub fn buffer(&self) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn buffer(&self, 
        _0: &mut Cmd,
        _1: StageFlags,
        _2: AccessFlags) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&self) -> &mut Image {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&self, 
        _0: &mut Cmd,
        _1: StageFlags,
        _2: AccessFlags) -> &mut Image {
        
        todo!();
        /*
        
        */
    }
    
    pub fn staging(&self) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn staging(&self, 
        _0: &mut Cmd,
        _1: StageFlags,
        _2: AccessFlags) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn fence(&self, _0: AccessFlags) -> &mut Fence {
        
        todo!();
        /*
        
        */
    }

    // Validation
    pub fn verify(&self)  {
        
        todo!();
        /*
        
        */
    }
}

/**
  | This class represents a Vulkan tensor and
  | provides an abstraction layer that allows both
  | the CPU, and the GPU, to view a Vulkan (buffer,
  | image) pair as one coherent, synchronized unit
  | of storage on both UMA and discrete systems.
  |
  | Expanding on the previous sentence, this class
  | tries to address two orthogonal implementation
  | complexities that arise as a result of the
  | aforementioned goal of memory coherence:
  |
  | 1) First, synchronization across processors;
  |    CPUs and GPUs are separate processors, and
  |    even though they share the same address
  |    space in a system with a unified memory
  |    architecture, their address spaces only
  |    partially overlap on systems with a discrete
  |    GPU.
  |
  |    Consequently on discrete systems, while it
  |    is still technically possible to take
  |    advantage of this shared address space to
  |    maintain one single copy of the data,
  |    different access latencies from CPU and GPU
  |    to this shared location usually necessitates
  |    maintaining two copies each in
  |    processor-local memory, otherwise memory
  |    access latency will hurt from the processor
  |    to which this data is not close.
  |
  |    This shared memory is more often than not
  |    located in system memory, making for slow
  |    GPU read and write access over the PCI-e bus
  |    on discrete. Maintaining two separate copies
  |    on the other hand, requires synchronization
  |    to guarantee coherence.  This is not an
  |    issue on UMA and this implementation
  |    accounts for that optimization.
  |
  | 2) Second, synchronization across resources
  |    (i.e. buffers and images); GPU drivers pack
  |    images in proprietory formats for better
  |    locality of access and to enable lossless
  |    compression.
  |
  |    These conversions are both expensive (in
  |    general) and manual (in Vulkan.)  This
  |    requires a second order of synchronization
  |    to guarantee coherence between the contents
  |    of the buffer and image otherwise they will
  |    go out of sync.
  |
  | It is extremely important to keep in mind that
  | the functionality this class provides is
  | generally expensive.  For optimal performance,
  | the user of this class should:
  |
  | 1) Avoid frequent CPU <=> GPU transfers which
  |    will be triggered if data is write accessed
  |    on one processor and read / write accessed
  |    on the other.
  |
  | 2) Avoid frequent buffer <=> image conversions
  |    which will be trigerred if data is write
  |    accessed as a buffer (image) and read
  |    accessed as an image (buffer).
  |
  | 3) When and if a synchronization is
  |    unavoidable, place as much distance between
  |    the synchronization is triggered and the
  |    data is accessed since all synchronizations
  |    this class provides are async.
  |
  | For optimal performance, access the data as
  | images, and keep the data on GPU, and above all
  | understand the expensive data flow that this
  | class abstracts away.
  |
  | VulkanTensor tries to address a specific concern and
  | intentionally does not expose GPU tensor memory
  | directly.  Please keep that behavior intact as
  | the whole data model fundamentally depends on
  | limiting what the user can achieve through the
  | interface to guarantee performance and
  | coherence.
  |
  | A VulkanTensor is associated with an api::Context as
  | preparation for multi-GPU support.
  |
  */
#[derive(Default)]
pub struct VulkanTensor {

    /**
      | Even at the cost of a heap allocation plus
      | the resulting negative impact on cache
      | locality due to the subsequent pointer
      | chasing, it is still critcal to share the
      | view across VulkanTensor implementations to
      | minimize programmer errors.
      |
      | Ideally this class should have been only made
      | movable, and non-copyable - something we
      | cannot do unfortunately due to the inner
      | workings of TensorImpl requiring copy
      | semantics in TensorImpl::release_resources()
      | to function as expected.
      |
      | Now that this class is made copyable though,
      | a new door to a whole new class of bugs is
      | opened, in that there now is a chance of two
      | [shallow] copies, have their State objects go
      | out of sync as a result of an operation being
      | performed on one shallow copy that is not
      | reflected in the other.
      |
      | Technically, if the programmer is very
      | careful, it is possible to avoid this trap
      | and not pay the cost of indirection, but the
      | resulting bugs of missing memory barriers
      | will be so frustrating to hunt down for those
      | unfamiliar with the internal mechanics of
      | this class, that I decided to take the
      | performance pentalty of this extra layer of
      | indirection in favor of making this class
      | easier to use.
      */
    view: Arc<View>,
}

impl VulkanTensor {
    
    pub fn new(
        context: *mut Context,
        sizes:   &[i32],
        options: &TensorOptions) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(
        context: *mut Context,
        pool:    *mut ResourcePool,
        sizes:   &[i32],
        options: &TensorOptions) -> Self {
    
        todo!();
        /*


        
        */
    }
}

impl HasStage for VulkanTensor {
    type Stage = PipelineStage;
}

impl HasAccess for VulkanTensor {
    type Access = ResourceMemoryAccess;
}

impl HasBuffer for VulkanTensor {
    type Buffer = ResourceBuffer;
}

impl HasFence for VulkanTensor {
    type Fence = ResourceFence;
}

impl HasImage for VulkanTensor {
    type Image = ResourceImage;
}

impl HasMemory for VulkanTensor {
    type Memory = ResourceMemory;
}

impl VulkanTensor {

    /**
      | Host access - these functions will be
      | expensive if they trigger a GPU -> CPU
      | sync due to pending writes.
      | 
      | A call to host() will trigger an async
      | copy in such scenarios, which is then
      | explicitly waited on as part of
      | 
      | Future::wait(). Consequently, for
      | optimal performance, put as much time
      | and distance between the place where
      | this function is called, and the location
      | where the future is waited on.
      |
      */
    pub fn host<Type>(&self, _0: &mut CommandBuffer) -> Future<Type,AccessRead> {
    
        todo!();
        /*
        
        */
    }
    
    pub fn host<Type, const kAccess: AccessFlags>(&mut self, _0: &mut CommandBuffer) -> Future<Type,kAccess> {
    
        todo!();
        /*
        
        */
    }

    /**
      | Device access - these functions will
      | be expensive if they trigger a buffer
      | <-> image or CPU -> GPU sync due to pending
      | writes.
      | 
      | These functions are non-blocking on
      | the host as the copy operation is carried
      | out by the
      | 
      | GPU asynchronously.
      | 
      | Regardless, they result in extra work
      | that could have been avoided or at least
      | minimized if all data access had occured
      | through one single processor (GPU in
      | this case) and on one type of resource
      | (image for best performance.)
      | 
      | Consequently, for optimal performance,
      | avoid mixed reads and writes across
      | processor boundaries, and do your best
      | to minimize layout transitions as a
      | result of working with images only (as
      | opposed to mixed buffer
      | 
      | - image usage.)
      | 
      | This implementation intentionally
      | restricts user access to the buffer
      | and image objects only, as opposed to
      | their underlying memory, for the sake
      | of predictability of usage and efficiency.
      |
      */
    pub fn buffer(&self, 
        _0: &mut CommandBuffer,
        _1: StageFlags) -> BufferObject {
        
        todo!();
        /*
        
        */
    }
    
    pub fn buffer(&mut self, 
        _0: &mut CommandBuffer,
        _1: StageFlags,
        _2: AccessFlags) -> BufferObject {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_image(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&self, 
        _0: &mut CommandBuffer,
        _1: StageFlags) -> ImageObject {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&mut self, 
        _0: &mut CommandBuffer,
        _1: StageFlags,
        _2: AccessFlags) -> ImageObject {
        
        todo!();
        /*
        
        */
    }

    pub fn extents(&self) -> &Uvec3 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn options(&self) -> &TensorOptions {
        
        todo!();
        /*
        
        */
    }
    
    pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }
    
    pub fn nbytes(&self) -> usize {
        
        todo!();
        /*
        
        */
    }
 
    // Some overloads below are intentionally
    // disabled to enforce a usage pattern that
    // ensures the Tensor's lifetime exceeds that of
    // the scope in which the underlying data is
    // accessed.  
    //
    // Allowing deleted overloads below to be
    // invoked on a temporary would open the door to
    // the possibility of accessing the underlying
    // memory out of the expected scope.

    /* ---------------------- Host  ---------------------- */
    pub fn host(&self, _0: &mut CommandBuffer) -> *const VulkanTensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn host(&mut self, 
        _0: &mut CommandBuffer,
        _1: AccessFlags) -> *mut VulkanTensor {
        
        todo!();
        /*
        
        */
    }
}

/* --------------------- Device  --------------------- */

pub type VulkanTensorImpl = VulkanOpaqueTensorImpl<VulkanTensor>;

impl Drop for VulkanTensorFuture {

    fn drop(&mut self) {
        todo!();
        /*
            #if VULKAN_SYNC_TENSORS_EAGERLY
      // Sync eagerly in an effort to hide latency.
      // Upside: Kick off the async transfer early on to keep the GPU busy.
      // Downside: An extra CPU command submission.
      if (tensor_ && (Access::Write & kAccess)) {
        if (tensor_->has_image()) {
          tensor_->image();
        }
        else {
          tensor_->buffer();
        }
      }
    #endif
        */
    }
}

impl VulkanTensorFuture {
    
    pub fn new(tensor: *const VulkanTensor) -> Self {
    
        todo!();
        /*


            : tensor_(tensor) 
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          tensor_,
          "Invalid Vulkan tensor!");
        */
    }
    
    pub fn new(future: Future) -> Self {
    
        todo!();
        /*


            : tensor_(move(future.tensor_)) 
      future.tensor_ = nullptr;
        */
    }
    
    #[inline] pub fn assign_from(&mut self, future: Future) -> &mut VulkanTensorFuture<Type,kAccess> {
        
        todo!();
        /*
            tensor_ = move(future.tensor_);
      future.tensor_ = nullptr;
      return *this;
        */
    }
    
    pub fn new<T, const A: AccessFlags>(future: Future<T,A>) -> Self {
    
        todo!();
        /*


            : tensor_(move(future.tensor_)) 
      future.tensor_ = nullptr;
        */
    }
    
    #[inline] pub fn assign_from(&mut self, future: Future<Type,kAccess>) -> &mut VulkanTensorFuture<Type,kAccess> {
        
        todo!();
        /*
            tensor_ = move(future.tensor_);
      future.tensor_ = nullptr;
      return *this;
        */
    }
    
    #[inline] pub fn wait(&self) -> VulkanTensorFuturePayload {
        
        todo!();
        /*
            TORCH_CHECK(
          tensor_,
          "VulkanTensorFuture is in an invalid state!  "
          "Potential reason: This future is moved from.");

      return tensor_->view_->wait().template map<Type, kAccess>();
        */
    }
}

impl VulkanTensor {
    
    #[inline] pub fn host(&self, command_buffer: &mut CommandBuffer) -> VulkanTensorFuture<Type,VulkanTensorAccessRead> {
        
        todo!();
        /*
            return Future<Type, VulkanTensorAccess::Read>(host(command_buffer));
        */
    }
    
    #[inline] pub fn host(&mut self, command_buffer: &mut CommandBuffer) -> VulkanTensorFuture<Type,kAccess> {
        
        todo!();
        /*
            return Future<Type, kAccess>(host(command_buffer, kAccess));
        */
    }
    
    #[inline] pub fn has_image(&self) -> bool {
        
        todo!();
        /*
            return view_->has_image();
        */
    }
    
    #[inline] pub fn extents(&self) -> &Uvec3 {
        
        todo!();
        /*
            return view_->extents();
        */
    }
    
    #[inline] pub fn options(&self) -> &TensorOptions {
        
        todo!();
        /*
            return view_->options();
        */
    }
    
    #[inline] pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
            return view_->sizes();
        */
    }
    
    #[inline] pub fn nbytes(&self) -> usize {
        
        todo!();
        /*
            return elementSize(typeMetaToScalarType(options().dtype())) *
             multiply_integers(sizes());
        */
    }
    
    #[inline] pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            return view_->strides();
        */
    }
}

impl VulkanTensorView {
    
    #[inline] pub fn has_image(&self) -> bool {
        
        todo!();
        /*
            return state_.is_available(View::Component::Image);
        */
    }
    
    #[inline] pub fn extents(&self) -> &Uvec3 {
        
        todo!();
        /*
            return extents_;
        */
    }
    
    #[inline] pub fn options(&self) -> &TensorOptions {
        
        todo!();
        /*
            return options_;
        */
    }
    
    #[inline] pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
            return sizes_;
        */
    }
    
    #[inline] pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            return strides_;
        */
    }
}

impl VulkanTensorViewStateBundleBuffer {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return (0u != stage) &&
             (0u != access);
        */
    }
}


impl VulkanTensorViewStateBundleImage {

    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return (0u != stage) &&
             (0u != access) &&
             (VK_IMAGE_LAYOUT_UNDEFINED != layout);
        */
    }
}

impl VulkanTensorViewState {
    
    #[inline] pub fn is_available(&self, components: ComponentFlags) -> bool {
        
        todo!();
        /*
            return available_ & components;
        */
    }
    
    #[inline] pub fn is_discrete(&self) -> bool {
        
        todo!();
        /*
            return is_available(Component::Staging);
        */
    }
    
    #[inline] pub fn is_uma(&self) -> bool {
        
        todo!();
        /*
            return !is_discrete();
        */
    }
    
    #[inline] pub fn is_clean(&self, components: ComponentFlags) -> bool {
        
        todo!();
        /*
            return !is_dirty(components);
        */
    }
    
    #[inline] pub fn is_dirty(&self, components: ComponentFlags) -> bool {
        
        todo!();
        /*
            return dirty_ & components;
        */
    }
    
    #[inline] pub fn set_clean(&mut self, components: ComponentFlags)  {
        
        todo!();
        /*
            dirty_ &= ~components;
        */
    }
    
    #[inline] pub fn set_dirty(&mut self, components: ComponentFlags)  {
        
        todo!();
        /*
            dirty_ |= components;
        */
    }
}

#[inline] pub fn convert(tensor: &Tensor) -> &mut VulkanTensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          tensor.is_vulkan(),
          "Vulkan tensor expected!");

      vTensorImpl* const impl =
          static_cast<vTensorImpl*>(tensor.unsafeGetTensorImpl());

      return impl->unsafe_opaque_handle();
        */
}

#[inline] pub fn convert_vulkan_tensor(tensor: &VulkanTensor) -> Tensor {
    
    todo!();
        /*
            return make_tensor<vTensorImpl>(
          DispatchKeySet(DispatchKey::Vulkan),
          tensor.options().dtype(),
          Device(kVulkan),
          tensor,
          tensor.sizes(),
          tensor.strides());
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Tensor.cpp]

pub fn vk_format(dtype: TypeMeta) -> VkFormat {
    
    todo!();
        /*
            switch (typeMetaToScalarType(dtype)) {
        case kFloat:
        #ifdef USE_VULKAN_FP16_INFERENCE
          return VK_FORMAT_R16G16B16A16_SFLOAT;
        #else
          return VK_FORMAT_R32G32B32A32_SFLOAT;
        #endif /* USE_VULKAN_FP16_INFERENCE */

        default:
          TORCH_CHECK(
              false,
              "Vulkan tensor format not supported!");
      }

      return VK_FORMAT_UNDEFINED;
        */
}

pub fn vk_extent(extent: &Uvec3) -> VkExtent3D {
    
    todo!();
        /*
            return {
        extent.data[0u],
        extent.data[1u],
        extent.data[2u],
      };
        */
}

pub fn access(vk_access: VkAccessFlags) -> VulkanTensorAccessFlags {
    
    todo!();
        /*
            VulkanTensorAccess::Flags access = 0u;

      constexpr VkAccessFlags kRead =
          VK_ACCESS_HOST_READ_BIT |
          VK_ACCESS_MEMORY_READ_BIT |
          VK_ACCESS_SHADER_READ_BIT |
          VK_ACCESS_TRANSFER_READ_BIT |
          VK_ACCESS_UNIFORM_READ_BIT;

      constexpr VkAccessFlags kWrite =
          VK_ACCESS_HOST_WRITE_BIT |
          VK_ACCESS_MEMORY_WRITE_BIT |
          VK_ACCESS_SHADER_WRITE_BIT |
          VK_ACCESS_TRANSFER_WRITE_BIT;

      if (vk_access & kRead) {
        access |= VulkanTensorAccess::Read;
      }

      if (vk_access & kWrite) {
        access |= VulkanTensorAccess::Write;
      }

      return access;
        */
}

pub fn vk_access(
    stage:  VulkanTensorStageFlags,
    access: VulkanTensorAccessFlags) -> VkAccessFlags {
    
    todo!();
        /*
            VkAccessFlags vk_access = 0u;

      if (access & VulkanTensorAccess::Read) {
        if (stage & VulkanTensorStage::Compute) {
          vk_access |= VK_ACCESS_SHADER_READ_BIT;
        }

        if (stage & VulkanTensorStage::Host) {
          vk_access |= VK_ACCESS_HOST_READ_BIT;
        }

        if (stage & VulkanTensorStage::Transfer) {
          vk_access |= VK_ACCESS_TRANSFER_READ_BIT;
        }
      }

      if (access & VulkanTensorAccess::Write) {
        if (stage & VulkanTensorStage::Compute) {
          vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
        }

        if (stage & VulkanTensorStage::Host) {
          vk_access |= VK_ACCESS_HOST_WRITE_BIT;
        }

        if (stage & VulkanTensorStage::Transfer) {
          vk_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
        }
      }

      return vk_access;
        */
}

pub fn vk_layout(
    stage:  VulkanTensorStageFlags,
    access: VulkanTensorAccessFlags) -> VkImageLayout {
    
    todo!();
        /*
            switch (stage) {
        case VulkanTensorStage::Compute:
          switch (access) {
            case VulkanTensorAccess::Read:
              return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            default:
              return VK_IMAGE_LAYOUT_GENERAL;
          } break;

        case VulkanTensorStage::Transfer:
          switch (access) {
            case VulkanTensorAccess::Read:
              return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            case VulkanTensorAccess::Write:
              return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

            default:
              TORCH_INTERNAL_ASSERT(false, "Invalid!");
          } break;

        default:
          TORCH_INTERNAL_ASSERT(false, "Invalid!");
      }

      return VK_IMAGE_LAYOUT_UNDEFINED;
        */
}

pub fn vk_stage(stage: VulkanTensorStageFlags) -> VkPipelineStageFlags {
    
    todo!();
        /*
            VkPipelineStageFlags vk_stage = 0u;

      if (stage & VulkanTensorStage::Compute) {
        vk_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      }

      if (stage & VulkanTensorStage::Host) {
        vk_stage |= VK_PIPELINE_STAGE_HOST_BIT;
      }

      if (stage & VulkanTensorStage::Transfer) {
        vk_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      }

      return vk_stage;
        */
}

pub fn buffer_bytes(
    sizes: &[i32],
    dtype: TypeMeta) -> VkDeviceSize {
    
    todo!();
        /*
            VkDeviceSize size = elementSize(typeMetaToScalarType(dtype));

      if (requires_image(sizes)) {
        const uvec3 extents = image_extents(sizes);
        size *= extents.data[0u] * extents.data[1u] * (4u * extents.data[2u]);
      }
      else {
        size *= multiply_integers(sizes);
      }

      return size;
        */
}

pub fn allocate_buffer(
    adapter: *const Adapter,
    pool:    *mut ResourcePool,
    sizes:   &[i32],
    options: &TensorOptions) -> VulkanTensorBuffer {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          adapter,
          "Invalid Vulkan adapter!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          pool,
          "Invalid Vulkan resource pool!");

      TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
      verify(options);

      const VkFlags usage =
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT;

      const auto memory = [adapter]() -> Resource::Memory::Descriptor {
        if (requires_staging(adapter)) {
          return {
            VMA_MEMORY_USAGE_GPU_ONLY,
            0u,
            0u,
          };
        }

        return {
          VMA_MEMORY_USAGE_GPU_TO_CPU,
          0u,
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
      }();

      return pool->buffer({
          buffer_bytes(sizes, options.dtype()),
          // Usage
          {
            usage,
            memory,
          },
        });
        */
}

pub fn requires_image(sizes: &[i32]) -> bool {
    
    todo!();
        /*
            return (1u <= sizes.size()) && (sizes.size() <= 4u);
        */
}

pub fn image_extents(sizes: &[i32]) -> Uvec3 {
    
    todo!();
        /*
            i64 width = 1;
      i64 height = 1;
      i64 depth = 1;

      switch (sizes.size()) {
        case 1:
          width = sizes[0];
          break;

        case 2:
          width = sizes[1];
          height = sizes[0];
          break;

        case 3:
          width = sizes[2];
          height = sizes[1];
          depth = sizes[0];
          break;

        case 4:
          width = sizes[3];
          height = sizes[2];
          depth = sizes[0] * sizes[1];
          break;

        default:
          TORCH_INTERNAL_ASSERT(
              false,
              "Only Tensors with 1 <= dim <= 4 can be represented as a Vulkan Image!");
      }

      return {
        safe_downcast<u32>(width),
        safe_downcast<u32>(height),
        safe_downcast<u32>(div_up(depth, INT64_C(4))),
      };
        */
}

pub fn allocate_image(
        pool:    *mut ResourcePool,
        extents: &VkExtent3D,
        options: &TensorOptions) -> VulkanTensorImage {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          pool,
          "Invalid Vulkan resource pool!");

      verify(options);

      return pool->image({
          extents.depth == 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D,
          vk_format(options.dtype()),
          extents,
          // Usage
          {
            VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_STORAGE_BIT,
            {
              VMA_MEMORY_USAGE_GPU_ONLY,
              0u,
              0u,
            },
          },
          // View
          {
            extents.depth == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D,
            vk_format(options.dtype()),
          },
          // Sampler
          {
            VK_FILTER_NEAREST,
            VK_SAMPLER_MIPMAP_MODE_NEAREST,
            VK_SAMPLER_ADDRESS_MODE_REPEAT,
            VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
          },
        });
        */
}

pub fn requires_staging(adapter: *const Adapter) -> bool {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          adapter,
          "Invalid Vulkan adapter!");

      return !adapter->has_unified_memory();
        */
}

pub fn allocate_staging(
        adapter: *const Adapter,
        pool:    *mut ResourcePool,
        sizes:   &[i32],
        options: &TensorOptions) -> VulkanTensorBuffer {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          adapter,
          "Invalid Vulkan adapter!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          pool,
          "Invalid Vulkan resource pool!");

      TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
      verify(options);

      return pool->buffer({
          buffer_bytes(sizes, options.dtype()),
          // Usage
          {
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            {
              VMA_MEMORY_USAGE_CPU_COPY,
              0u,
              0u,
            },
          },
        });
        */
}

pub fn allocate_fence(pool: *mut ResourcePool) -> VulkanTensorFence {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          pool,
          "Invalid Vulkan resource pool!");

      return pool->fence();
        */
}

pub enum Barrier {
    None,
    Exectution,
    Memory,
}

pub fn categorize_a(
    vk_src_access: VkAccessFlags,
    vk_dst_access: VkAccessFlags) -> Barrier {
    
    todo!();
        /*
            if (0u == vk_src_access) {
        return Barrier::None;
      }

      const VulkanTensorAccess::Flags src_access = access(vk_src_access);
      const VulkanTensorAccess::Flags dst_access = access(vk_dst_access);

      if ((src_access & VulkanTensorAccess::Read) == src_access) {
        if ((dst_access & VulkanTensorAccess::Read) == dst_access) {
          // RAR (Read after Read)
          return Barrier::None;
        }

        // WAR (Write after Read)
        return Barrier::Exectution;
      }

      // RAW (Read after Write), or WAW (Write after Write)
      return Barrier::Memory;
        */
}

pub fn categorize_b(
    vk_src_access: VkAccessFlags,
    vk_dst_access: VkAccessFlags,
    vk_src_layout: VkImageLayout,
    vk_dst_layout: VkImageLayout) -> Barrier {

    todo!();
    /*
            if (vk_src_layout != vk_dst_layout) {
        return Barrier::Memory;
      }

      return categorize(vk_src_access, vk_dst_access);
        */
}

impl VulkanTensor {
    
    pub fn new(
        context: *mut Context,
        sizes:   &[i32],
        options: &TensorOptions) -> Self {
    
        todo!();
        /*


            : VulkanTensor(
          context,
          &context->resource().pool,
          sizes,
          options)
        */
    }
    
    pub fn new(
        context: *mut Context,
        pool:    *mut ResourcePool,
        sizes:   &[i32],
        options: &TensorOptions) -> Self {
    
        todo!();
        /*


            : view_(new View{
          context,
          pool,
          sizes,
          options,
        })
        */
    }
}


impl VulkanTensor {
    
    pub fn host(&self, command_buffer: &mut CommandBuffer) -> *const VulkanTensor {
        
        todo!();
        /*
            view_->staging(command_buffer, Stage::Host, Access::Read);
      return this;
        */
    }
    
    pub fn host(&mut self, 
        command_buffer: &mut CommandBuffer,
        access:         AccessFlags) -> *mut VulkanTensor {
        
        todo!();
        /*
            view_->staging(command_buffer, Stage::Host, access);
      return this;
        */
    }
    
    pub fn buffer(&self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags) -> VulkanTensorBufferObject {
        
        todo!();
        /*
            return view_->buffer(
          command_buffer,
          stage,
          Access::Read).object;
        */
    }
    
    pub fn buffer(&mut self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags,
        access:         AccessFlags) -> VulkanTensorBufferObject {
        
        todo!();
        /*
            return view_->buffer(
          command_buffer,
          stage,
          access).object;
        */
    }
    
    pub fn image(&self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags) -> VulkanTensorImageObject {
        
        todo!();
        /*
            return view_->image(
          command_buffer,
          stage,
          Access::Read).object;
        */
    }
    
    pub fn image(&mut self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags,
        access:         AccessFlags) -> VulkanTensorImageObject {
        
        todo!();
        /*
            return view_->image(
          command_buffer,
          stage,
          access).object;
        */
    }
}


impl VulkanTensorView {
    
    pub fn new() -> Self {
    
        todo!();
        /*


            // Resources
      : buffer_{},
        image_{},
        staging_{},
        fence_{},
        // Context
        context_(nullptr),
        pool_(nullptr),
        // State
        state_{},
        // Metadata
        extents_{}
        */
    }
    
    pub fn new(
        context: *mut Context,
        pool:    *mut ResourcePool,
        sizes:   &[i32],
        options: &TensorOptions) -> Self {
    
        todo!();
        /*


            // Resources
      : buffer_{},
        image_{},
        staging_{},
        fence_{},
        // Context
        context_(context),
        pool_(pool),
        // State
        state_(context->gpu().adapter, sizes),
        // Metadata
        extents_(image_extents(sizes)),
        options_(options),
        sizes_(sizes),
        strides_(sizes.size()) 
      ops::verify(options);
        */
    }
}

pub struct VulkanTensorViewCmd {
    view:           &View,
    command_buffer: &mut CommandBuffer,
}

impl HasBuffer for VulkanTensorViewCmd {
    type Buffer = ResourceBuffer;
}

impl HasImage for VulkanTensorViewCmd {
    type Image = ResourceImage;
}

impl HasFence for VulkanTensorViewCmd {
    type Fence = ResourceFence;
}

impl VulkanTensorViewCmd {
    
    pub fn new(
        view:           &View,
        command_buffer: &mut CommandBuffer) -> Self {
    
        todo!();
        /*


            : view_(view),
        command_buffer_(command_buffer)
        */
    }
    
    pub fn barrier(&mut self, transition: StateTransition)  {
        
        todo!();
        /*
            // Buffer and Staging are just an alias for the same memory region on UMA.

      if (view_.state_.is_uma()) {
        transition.first.buffer.stage |= transition.first.staging.stage;
        transition.first.buffer.access |= transition.first.staging.access;
        transition.first.staging = {};

        transition.second.buffer.stage |= transition.second.staging.stage;
        transition.second.buffer.access |= transition.second.staging.access;
        transition.second.staging = {};
      }

      // Filter out host dependencies out of source, per Vulkan spec host write ordering guarantees:
      // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes

      const auto filter_stage =[](VkPipelineStageFlags& stage) {
        stage &= ~VK_PIPELINE_STAGE_HOST_BIT;
      };

      filter_stage(transition.first.buffer.stage);
      filter_stage(transition.first.staging.stage);

      const auto filter_access =[](VkAccessFlags& access) {
        access &= ~(VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT);
      };

      filter_access(transition.first.buffer.access);
      filter_access(transition.first.staging.access);

      Pipeline::Barrier barrier{};

      if (transition.second.staging) {
        const State::Bundle::Buffer from = transition.first.staging;
        const State::Bundle::Buffer to = transition.second.staging;

        const Barrier category = categorize(
            from.access,
            to.access);

        if (Barrier::None != category) {
          barrier.stage.src |= from.stage;
          barrier.stage.dst |= to.stage;

          if (Barrier::Memory == category) {
            barrier.buffers.push_back({
              view_.staging().object,
              {
                from.access,
                to.access,
              },
            });
          }
        }
      }

      if (transition.second.buffer) {
        const State::Bundle::Buffer from = transition.first.buffer;
        const State::Bundle::Buffer to = transition.second.buffer;

        const Barrier category = categorize(
            from.access,
            to.access);

        if (Barrier::None != category) {
          barrier.stage.src |= from.stage;
          barrier.stage.dst |= to.stage;

          if (Barrier::Memory == category) {
            barrier.buffers.push_back({
              view_.buffer().object,
              {
                from.access,
                to.access,
              },
            });
          }
        }
      }

      if (transition.second.image) {
        const State::Bundle::Image from = transition.first.image;
        const State::Bundle::Image to = transition.second.image;

        const Barrier category = categorize(
            from.access,
            to.access,
            from.layout,
            to.layout);

        if (Barrier::None != category) {
          barrier.stage.src |= from.stage;
          barrier.stage.dst |= to.stage;

          if (Barrier::Memory == category) {
            TORCH_INTERNAL_ASSERT(
                from.layout == view_.image().object.layout,
                "Invalid image layout!");

            barrier.images.push_back({
              view_.image().object,
              {
                from.access,
                to.access,
              },
              {
                from.layout,
                to.layout,
              },
            });

            view_.image().object.layout = to.layout;
          }
        }
      }

      // If we are left with anything meaningful, insert a barrier.

      if (barrier) {
        if (0u == barrier.stage.src) {
          barrier.stage.src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }

        if (0u == barrier.stage.dst) {
          barrier.stage.src = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        }

        command_buffer_.barrier(barrier);
      }
        */
    }
    
    pub fn copy_buffer_to_staging(&mut self, 
        state:   &mut State,
        buffer:  &BufferObject,
        staging: &mut BufferObject)  {
        
        todo!();
        /*
            if (state.is_clean(Component::Staging) || state.is_uma()) {
        return;
      }

      barrier(
          state.transition({
              // Staging
              {
                vk_stage(Stage::Transfer),
                vk_access(Stage::Transfer, Access::Write),
              },
              // Buffer
              {
                vk_stage(Stage::Transfer),
                vk_access(Stage::Transfer, Access::Read),
              },
              // Image
              {},
            }));

      command_buffer_.copy(buffer, staging);
        */
    }
    
    pub fn copy_staging_to_buffer(&mut self, 
        state:   &mut State,
        staging: &BufferObject,
        buffer:  &mut BufferObject)  {
        
        todo!();
        /*
            if (state.is_clean(Component::Buffer) || state.is_uma()) {
        return;
      }

      barrier(
          state.transition({
              // Staging
              {
                vk_stage(Stage::Transfer),
                vk_access(Stage::Transfer, Access::Read),
              },
              // Buffer
              {
                vk_stage(Stage::Transfer),
                vk_access(Stage::Transfer, Access::Write),
              },
              // Image
              {},
            }));

      command_buffer_.copy(staging, buffer);
        */
    }
    
    pub fn copy_buffer_to_image(&mut self, 
        state:  &mut State,
        buffer: &BufferObject,
        image:  &mut ImageObject)  {
        
        todo!();
        /*
            if (state.is_clean(Component::Image)) {
        return;
      }

      barrier(
          state.transition({
              // Staging
              {},
              // Buffer
              {
                vk_stage(Stage::Compute),
                vk_access(Stage::Compute, Access::Read),
              },
              // Image
              {
                vk_stage(Stage::Compute),
                vk_access(Stage::Compute, Access::Write),
                vk_layout(Stage::Compute, Access::Write),
              },
            }));

      const uvec3 extents = view_.extents();
      const u32 plane = extents.data[0u] * extents.data[1u];

      const struct Block final {
        uvec3 extents;
        u32 block;
        uvec4 offset;
      } block {
        extents,
        4u * plane,
        {
          0u * plane,
          1u * plane,
          2u * plane,
          3u * plane,
        },
      };

      view_.context_->dispatch(
          command_buffer_,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(nchw_to_image),
          extents,
          view_.context_->gpu().adapter->local_work_group_size(),
          image,
          buffer,
          view_.context_->resource().pool.uniform(block).object);
        */
    }
    
    pub fn copy_image_to_buffer(&mut self, 
        state:  &mut State,
        image:  &ImageObject,
        buffer: &mut BufferObject)  {
        
        todo!();
        /*
            if (state.is_clean(Component::Buffer)) {
        return;
      }

      barrier(
          state.transition({
              // Staging
              {},
              // Buffer
              {
                vk_stage(Stage::Compute),
                vk_access(Stage::Compute, Access::Write),
              },
              // Image
              {
                vk_stage(Stage::Compute),
                vk_access(Stage::Compute, Access::Read),
                vk_layout(Stage::Compute, Access::Read),
              },
            }));

      const uvec3 extents = view_.extents();
      const u32 plane = extents.data[0u] * extents.data[1u];

      const struct Block final {
        uvec3 extents;
        u32 block;
        uvec4 offset;
      } block {
        extents,
        4u * plane,
        {
          0u * plane,
          1u * plane,
          2u * plane,
          3u * plane,
        },
      };

      view_.context_->dispatch(
          command_buffer_,
          {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(image_to_nchw),
          view_.extents(),
          view_.context_->gpu().adapter->local_work_group_size(),
          image,
          buffer,
          view_.context_->resource().pool.uniform(block).object);
        */
    }
    
    pub fn submit(&mut self, fence: ResourceFence)  {
        
        todo!();
        /*
            view_.context_->command().pool.submit(
          view_.context_->gpu().queue,
          command_buffer_,
          fence);
        */
    }
}

impl VulkanTensorView {
    
    pub fn buffer(&self) -> &mut VulkanTensorBuffer {
        
        todo!();
        /*
            if (!buffer_) {
        buffer_ = allocate_buffer(
            context_->gpu().adapter,
            pool_,
            sizes(),
            options());
      }

      return buffer_;
        */
    }
    
    pub fn buffer(&self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags,
        access:         AccessFlags) -> &mut VulkanTensorBuffer {
        
        todo!();
        /*
            CMD cmd(*this, command_buffer);
      return buffer(cmd, stage, access);
        */
    }
    
    pub fn buffer(&self, 
        cmd:    &mut Cmd,
        stage:  StageFlags,
        access: AccessFlags) -> &mut VulkanTensorBuffer {
        
        todo!();
        /*
            if ((access & Access::Read) && state_.is_dirty(Component::Buffer)) {
        if (state_.is_clean(Component::Staging)) {
          cmd.copy_staging_to_buffer(
              state_,
              staging(cmd, Stage::Transfer, Access::Read).object,
              buffer().object);
        }
        else if (state_.is_clean(Component::Image)) {
          cmd.copy_image_to_buffer(
              state_,
              image(cmd, Stage::Compute, Access::Read).object,
              buffer().object);
        }
        else {
          TORCH_INTERNAL_ASSERT(
              false,
              "Invalid state!");
        }
      }

      cmd.barrier(
          state_.transition({
              // Staging
              {},
              // Buffer
              {
                vk_stage(stage),
                vk_access(stage, access),
              },
              // Image
              {},
            }));

      if (access & Access::Write) {
        state_.set_dirty(Component::All);
      }

      state_.set_clean(Component::Buffer);

      return buffer();
        */
    }
    
    pub fn image(&self) -> &mut VulkanTensorImage {
        
        todo!();
        /*
            if (!image_ && state_.is_available(Component::Image)) {
        image_ = allocate_image(
            pool_,
            vk_extent(extents()),
            options());
      }

      return image_;
        */
    }
    
    pub fn image(&self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags,
        access:         AccessFlags) -> &mut VulkanTensorImage {
        
        todo!();
        /*
            CMD cmd(*this, command_buffer);
      return image(cmd, stage, access);
        */
    }
    
    pub fn image(&self, 
        cmd:    &mut Cmd,
        stage:  StageFlags,
        access: AccessFlags) -> &mut VulkanTensorImage {
        
        todo!();
        /*
            if ((access & Access::Read) && state_.is_dirty(Component::Image)) {
        cmd.copy_buffer_to_image(
            state_,
            buffer(cmd, stage, Access::Read).object,
            image().object);
      }

      cmd.barrier(
          state_.transition({
              // Staging
              {},
              // Buffer
              {},
              // Image
              {
                vk_stage(stage),
                vk_access(stage, access),
                vk_layout(stage, access),
              },
            }));

      if (access & Access::Write) {
        state_.set_dirty(Component::All);
      }

      state_.set_clean(Component::Image);

      return image();
        */
    }
    
    pub fn staging(&self) -> &mut VulkanTensorBuffer {
        
        todo!();
        /*
            if (!state_.is_available(Component::Staging)) {
        return buffer();
      }

      if (!staging_) {
        staging_ = allocate_staging(
            context_->gpu().adapter,
            pool_,
            sizes(),
            options());
      }

      return staging_;
        */
    }
    
    pub fn staging(&self, 
        command_buffer: &mut CommandBuffer,
        stage:          StageFlags,
        access:         AccessFlags) -> &mut VulkanTensorBuffer {
        
        todo!();
        /*
            CMD cmd(*this, command_buffer);
      Buffer& staging = this->staging(cmd, stage, access);
      cmd.submit(fence(access));

      return staging;
        */
    }
    
    pub fn staging(&self, 
        cmd:    &mut Cmd,
        stage:  StageFlags,
        access: AccessFlags) -> &mut VulkanTensorBuffer {
        
        todo!();
        /*
            if ((access & Access::Read) && state_.is_dirty(Component::Staging)) {
        cmd.copy_buffer_to_staging(
            state_,
            buffer(cmd, Stage::Transfer, Access::Read).object,
            staging().object);
      }

      cmd.barrier(
          state_.transition({
              // Staging
              {
                vk_stage(stage),
                vk_access(stage, access),
              },
              // Buffer
              {},
              // Image
              {},
            }));

      if (access & Access::Write) {
        state_.set_dirty(Component::All);
      }

      state_.set_clean(Component::Staging);

      return staging();
        */
    }
    
    pub fn fence(&self, access: AccessFlags) -> &mut VulkanTensorFence {
        
        todo!();
        /*
            if (access & Access::Read) {
        fence_ = allocate_fence(&context_->resource().pool);
      }

      return fence_;
        */
    }
    
    pub fn wait(&self) -> &mut VulkanTensorMemory {
        
        todo!();
        /*
            if (fence_) {
        fence_.wait();
      }

      return staging().memory;
        */
    }
    
    pub fn verify(&self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(!image_ || state_.is_available(Component::Image));
      TORCH_INTERNAL_ASSERT(!staging_ || state_.is_discrete());
        */
    }
}

impl VulkanTensorViewState {
    
    pub fn new() -> Self {
    
        todo!();
        /*


            : available_{},
        dirty_{},
        bundle_{}
        */
    }
    
    pub fn new(
        adapter: *const Adapter,
        sizes:   &[i32]) -> Self {
    
        todo!();
        /*


            : available_{},
        dirty_{},
        bundle_{} 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          adapter,
          "Invalid Vulkan adapter!");

      available_ |= Component::Buffer;

      if (requires_image(sizes)) {
        available_ |= Component::Image;
      }

      if (requires_staging(adapter)) {
        available_ |= Component::Staging;
      }
        */
    }
    
    pub fn transition(&mut self, bundle: Bundle) -> VulkanTensorViewStateTransition {
        
        todo!();
        /*
            const Bundle from = bundle_;
      Bundle& to = bundle_;

      if (bundle.staging) {
        to.staging = bundle.staging;
      }

      if (bundle.buffer) {
        to.buffer = bundle.buffer;
      }

      if (bundle.image) {
        to.image = bundle.image;
      }

    #ifdef VULKAN_TENSOR_DEBUG
      cout << "From:" << endl << from << endl;
      cout << "To:" << endl << to << endl;
    #endif /* VULKAN_TENSOR_DEBUG */

      return Transition{
        from,
        to,
      };
        */
    }
}

pub fn verify(options: &TensorOptions)  {
    
    todo!();
        /*
            TORCH_CHECK(
          !options.has_requires_grad() || !options.requires_grad(),
          "'requires_grad' tensor option is not yet supported under Vulkan!");

      TORCH_CHECK(
          !options.has_pinned_memory() || !options.pinned_memory(),
          "'pinned_memory' tensor option is not yet supported under Vulkan!");

      TORCH_CHECK(
          !options.has_layout() || (kStrided == options.layout()),
          "'layout' tensor option is not yet supported under Vulkan!");

      TORCH_CHECK(
          !options.has_memory_format() ||
              (MemoryFormat::Contiguous == options.memory_format_opt()),
          "'memory_format' tensor option is not yet supported under Vulkan!");
        */
}


/**
  | Considering that VkAccessFlags is a weak
  | typedef of a built-in data type, we need to
  | introduce a new type to allow overload
  | resolution distinguish between the two.
  */
#[cfg(VULKAN_TENSOR_DEBUG)]
pub struct Access {
    value: VkAccessFlags,
}

#[cfg(VULKAN_TENSOR_DEBUG)]
impl fmt::Display for Access {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << "Access: ";

      if (0u == access.value) {
        return stream << "  0";
      }

      if (access.value & VK_ACCESS_HOST_READ_BIT) {
        stream << "  VK_ACCESS_HOST_READ_BIT";
      }

      if (access.value & VK_ACCESS_HOST_WRITE_BIT) {
        stream << "  VK_ACCESS_HOST_WRITE_BIT";
      }

      if (access.value & VK_ACCESS_MEMORY_READ_BIT) {
        stream << "  VK_ACCESS_MEMORY_READ_BIT";
      }

      if (access.value & VK_ACCESS_MEMORY_WRITE_BIT) {
        stream << "  VK_ACCESS_MEMORY_WRITE_BIT";
      }

      if (access.value & VK_ACCESS_SHADER_READ_BIT) {
        stream << "  VK_ACCESS_SHADER_READ_BIT";
      }

      if (access.value & VK_ACCESS_SHADER_WRITE_BIT) {
        stream << "  VK_ACCESS_SHADER_WRITE_BIT";
      }

      if (access.value & VK_ACCESS_TRANSFER_READ_BIT) {
        stream << "  VK_ACCESS_TRANSFER_READ_BIT";
      }

      if (access.value & VK_ACCESS_TRANSFER_WRITE_BIT) {
        stream << "  VK_ACCESS_TRANSFER_WRITE_BIT";
      }

      return stream;
        */
    }
}

/**
  | Considering that VkImageLayout is a weak
  | typedef of a built-in data type, we need to
  | introduce a new type to allow overload
  | resolution distinguish between the two.
  */
#[cfg(VULKAN_TENSOR_DEBUG)]
pub struct ImageLayout {
    value: VkImageLayout,
}

#[cfg(VULKAN_TENSOR_DEBUG)]
impl fmt::Display for ImageLayout {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << "Layout: ";

      switch (layout.value) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
          stream << "  VK_IMAGE_LAYOUT_UNDEFINED";
          break;

        case VK_IMAGE_LAYOUT_GENERAL:
          stream << "  VK_IMAGE_LAYOUT_GENERAL";
          break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
          stream << "  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
          break;

        default:
          stream << "  Unknown!";
          break;
      };

      return stream;
        */
    }
}

/**
  | Considering that VkPipelineStageFlags is a weak
  | typedef of a built-in data type, we need to
  | introduce a new type to allow overload
  | resolution distinguish between the two.
  */
#[cfg(VULKAN_TENSOR_DEBUG)]
pub struct Stage {
    value: VkPipelineStageFlags,
}

#[cfg(VULKAN_TENSOR_DEBUG)]
impl fmt::Display for Stage {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << "Stage: ";

      if (0u == stage.value) {
        return stream << "  0";
      }

      if (stage.value & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        stream << "  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT";
      }

      if (stage.value & VK_PIPELINE_STAGE_HOST_BIT) {
        stream << "  VK_PIPELINE_STAGE_HOST_BIT";
      }

      if (stage.value & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) {
        stream << "  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT";
      }

      if (stage.value & VK_PIPELINE_STAGE_TRANSFER_BIT) {
        stream << "  VK_PIPELINE_STAGE_TRANSFER_BIT";
      }

      return stream;
        */
    }
}

#[cfg(VULKAN_TENSOR_DEBUG)]
impl fmt::Display for VulkanTensorViewStateBundle {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << "Staging\n " <<
          Stage{
            bundle.staging.stage,
          } << "\n " <<
          Access{
            bundle.staging.access,
          } << endl;

      stream << "Buffer\n " <<
          Stage{
            bundle.buffer.stage,
          } << "\n " <<
          Access{
            bundle.buffer.access,
          } << endl;

      stream << "Image\n " <<
          Stage{
            bundle.image.stage,
          } << "\n " <<
          Access{
            bundle.image.access,
          } <<  "\n " <<
          Image::Layout{
            bundle.image.layout,
          } << endl;

      return stream;
        */
    }
}
