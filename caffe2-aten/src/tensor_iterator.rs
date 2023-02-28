/*!
  | TensorIterator is a helper class for
  | element-wise operations, such as arithmetic,
  | comparisons, and trigonometric functions. It
  | handles broadcasting and type conversions of
  | operands.
  |
  | This is inspired by NumPy's Array Iterator API
  | (NpyIter).
  |
  | The files Loops.h and Loops.cuh provide
  | functions to build kernels that use
  | TensorIterator.
  |
  | Example:
  |
  |   auto iter = TensorIteratorConfig()
  |     .add_output(output)
  |     .add_input(input)
  |     .build()
  |
  | [MyKernel.cpp / MyKernel.cu]
  |   cpu_kernel(iter, [](float a, float b) {
  |     return a + b;
  |   });
  |
  |   gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {
  |     return a + b;
  |   });
  |
  | Note [Order of Construction]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | When setting up the tensor iterator
  | configuration, the output Tensors have to be
  | added first via
  | TensorIteratorConfig::add_owned_output(Tensor).
  |
  | After adding all outputs, the inputs can be
  | added via
  | TensorIteratorConfig::add_owned_input(Tensor).
  |
  | Adding another output after inputs have been
  | added will rise an exception.
  |
  | Note [Common Dtype Computation]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Some operations have a natural notion of
  | a "common dtype" or "computation dtype" where
  | all inputs are cast to one dtype, the operation
  | is performed, and then the results are cast to
  | all outputs.
  |
  | TensorIterator infers a common dtype if all
  |   inputs have the same dtype, and it computes
  |   one using type promotion rules on its inputs
  |   if promote_inputs_to_common_dtype_ is true.
  |
  |   Attempting to query a common dtype otherwise
  |   will throw an exception.
  |
  | Note that the outputs are not considered when
  | computing a common dtype.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorIterator.h]

/**
  | This parameter is heuristically chosen to
  | determine the minimum number of work that
  | warrants parallelism.
  |
  | For example, when summing an array, it is
  | deemed inefficient to parallelise over arrays
  | shorter than 32768.
  |
  | Further, no parallel algorithm (such as
  | parallel_reduce) should split work into smaller
  | than GRAIN_SIZE chunks.
  |
  */
pub const GRAIN_SIZE: i64 = 32768;

#[derive(Default)]
pub struct OperandInfo {

    /**
      | Stride after broadcasting. The stride
      | is in bytes, not number of elements.
      |
      */
    stride_bytes:    StrideVector,

    /**
      | The tensor operand. Note that the strides,
      | data pointer, and other attributes
      | may differ due to dimension reordering
      | and coalescing.
      |
      */
    tensor:          MaybeOwned<Tensor>,

    /**
      | Save the original tensor operand in
      | cases when an output is modified (e.g.
      | if dtype is changed) 
      |
      | default = MaybeOwned<Tensor>::owned(in_place);
      |
      */
    original_tensor: MaybeOwned<Tensor>,

    /**
      | The desired device and type for the operand.
      | For inputs, this specifies that the
      | input should be converted to this type
      | if necessary. For outputs, this specifies
      | which type to allocate. target_dtype
      | and device are initialized with the
      | dtype and device of the tensor but during
      | type promotion target_dtype value
      | can become different from tensor's
      | dtype also, during type promotion target_dtype
      | and device can be set for an undefined
      | tensor so that tensor can be properly
      | constructed later.
      |
      */
    device:          Device, // default = kCPU

    /**
      | default = ScalarType::Undefined;
      |
      */
    target_dtype:    ScalarType,

    /**
      | Caches dtype of the tensor, because
      | scalar_type is an expensive operation
      | 
      | If dtype of the tensor is changed (e.g.
      | as a result of type promotion or in allocate_outputs),
      | this value should be changed too. = ScalarType::Undefined;
      |
      */
    current_dtype:   ScalarType,

    /**
      | The data pointer. This may be different
      | from tensor->data_ptr() if the iterator
      | is split.
      |
      */
    data:            *mut c_void, // default = nullptr

    is_output:       bool, // default = false
    will_resize:     bool, // default = false
    is_read_write:   bool, // default = false
}

pub mod operand_info {

    use super::*;

    pub type StrideVector = SmallVector<i64,6>;
}

impl OperandInfo {

    #[inline(always)]
    pub fn new(t: MaybeOwned<Tensor>) -> Self {
    
        todo!();
        /*
        : tensor(move(t)),

            if (tensor->defined()) {
          device = tensor->device();
          target_dtype = tensor->scalar_type();
          current_dtype = target_dtype;
        }
        validate();
        */
    }
    
    pub fn is_type_defined(&self) -> bool {
        
        todo!();
        /*
            return target_dtype != ScalarType::Undefined;
        */
    }
    
    pub fn options(&self) -> TensorOptions {
        
        todo!();
        /*
            return TensorOptions(target_dtype).device(device);
        */
    }
    
    pub fn validate(&mut self)  {
        
        todo!();
        /*
            TORCH_CHECK(
            !tensor->defined() || tensor->layout() == kStrided,
            "unsupported tensor layout: ", tensor->layout());
        */
    }
}

#[repr(u8)]
pub enum FastSetupType {
    NONE,
    CONTIGUOUS,
    CHANNELS_LAST,
    NON_OVERLAPPING_DENSE
}

pub struct TensorIteratorBase {

    base: MetaBase,

    /**
      | Records the "computation" shape of
      | the output tensor. The computation
      | shape is different from the regular
      | shape in a few ways:
      | 
      | - The shape may be permuted (via permute_dimensions)
      | so that we process the dimensions in
      | the most computationally efficient
      | order (rather than the logical order
      | given to us by the users.)
      | 
      | - The shape may have adjacent dimensions
      | collapsed (via coalesce_dimensions)
      | so that we minimize the number of dimensions
      | we have to explicitly iterate over.
      | For example, a pointwise operation
      | on a contiguous tensor "computationally"
      | consists of only a single dimension.
      | 
      | In other words, the computation shape
      | is the output shape as it actually matters
      | for implementing the kernel, but not
      | necessarily the output shape that the
      | user will see in the end.
      | 
      | The lifecycle of mutations to shape_
      | in
      | 
      | TensorIterator:
      | 
      | - declare_static_shape() sets an initial
      | shape explicitly provided by user,
      | otherwise
      | 
      | - compute_shape() computes the true
      | (non-computational) shape specified
      | by the user.
      | 
      | - reorder_dimensions() reorders dimensions
      | to improve coalescing.
      | 
      | - coalesce_dimensions() then coalesces
      | adjacent dimensions when possible.
      | 
      | The shape may also be further modified
      | if we create sub-TensorIterators,
      | e.g., via narrow or select_all_keeping_dim.
      |
      */
    shape:                    DimVector,

    /**
      | Temporarily records the permutation
      | computed by reorder_dimensions.
      | 
      | This permutation maps the computation
      | output dimension (dim) to the original
      | true output dimension (perm_[dim]).
      | 
      | It is used by invert_perm to undo the
      | permutation.
      | 
      | After coalesce_dimensions is called,
      | the permutation is no longer valid (as,
      | in general, there is no permutation
      | that will make computation dimensions
      | to output dimensions); methods that
      | manipulate perm_ are obligated to test
      | that !has_coalesced_dimensions
      |
      */
    perm:                     DimVector,

    /**
      | Has coalesce_dimensions() (or any
      | moral equivalent, e.g., fast_build())
      | been called?
      | 
      | This is SOLELY used to check validity
      | of perm_.
      |
      */
    has_coalesced_dimensions: bool, // default = false

    /**
      | Whether iteration must be fixed.
      | 
      | This disables dimension permuting
      | and also changes how for_each divides
      | work among threads.
      |
      */
    enforce_linear_iteration: bool, // default = false

    /**
      | The index offsets into the original
      | tensors for each dimension.
      | 
      | This is only non-zero when you narrow()
      | a TensorIterator (e.g., when you make
      | sub-TensorIterators).
      |
      */
    view_offsets:             DimVector,

    /**
      | The computed names of the output tensor.
      | 
      | Computed by compute_names()
      |
      */
    names:                    NameVector,

    /**
      | The operands of the TensorIterator:
      | both the inputs and outputs. The outputs
      | MUST come first in the operands_ list.
      | 
      | There is always an operand for each output
      | of the TensorIterator, even if
      | 
      | TensorIterator will ultimately be
      | responsible for allocating the output;
      | in those cases, tensor is simply undefined
      | (and will be populated later during
      | build()).
      | 
      | This list is initially populated prior
      | to build(), but build() mutates OperandInfo
      | to populate more information.
      |
      */
    operands:                 SmallVector<OperandInfo,4>,

    /**
      | Number of outputs in operands_ (the
      | length of the outputs prefix in operands_).
      |
      */
    num_outputs:              i32, // default = 0

    /**
      | Whether or not all operands have the
      | same shape. Having all the same shape
      | affects whether or not the iterator
      | is eligible for fast setup.
      |
      */
    all_ops_same_shape:       bool, // default = false

    /**
      | The "computation" dtype of TensorIterator,
      | specifying what the dtype we will do
      | the internal computation in TensorIterator.
      | Typically, this matches the dtype of
      | the output tensors, but not always!
      |
      */
    common_dtype:             ScalarType, // default = ScalarType_Undefined

    /**
      | This is currently defined as kCPU, or
      | the device of the first non-CPU tensor
      | argument. See TensorIteratorBase::compute_types
      | for details.
      |
      */
    common_device:            Device, // default = kCPU

    /**
      | Set by split(), see should_accumulate()
      | and is_final_output()
      |
      */
    accumulate:               bool, // default = false

    final_output:             bool, // default = true

    /**
      | From TensorIteratorConfig
      |
      */
    is_reduction:             bool, // default = false

    /**
      | Set by populate_operands(), says if
      | we're handling meta tensors
      |
      */
    is_meta:                  bool, // default = false
}

pub mod tensor_iterator_base {

    use super::*;

    pub type DimMask      = BitSet<64>;
    pub type PtrVector    = SmallVector<*mut u8,4>;
    pub type StrideVector = SmallVector<i64,6>;

    /**
      | The inner-loop function operates on the fastest
      |   moving dimension. It implements element-wise
      |   operations in terms of 1-d strided tensors.
      |
      | Arguments:
      |
      |  data: data pointers for each operand (length
      |  `ntensors`)
      |
      |  strides: stride for each operand (length
      |  `ntensors`)
      |
      |  size: size of inner loop
      |
      | The `size` often matches shape[0], but may be
      | smaller due to parallelization of the inner
      | loop.
      |
      */
    pub type Loop2d = fn(
        data:    *mut *mut u8,
        strides: *const i64,
        size0:   i64,
        size1:   i64
    ) -> ();

    pub type LoopSubiter = fn(subiter: &mut TensorIteratorBase) -> ();
}

#[macro_export] macro_rules! torch_disallow_temporaries_impl {
    ($methodname:ident, $maybestatic:ident) => {
        /*
        
          maybestatic void methodname(Tensor&& out, const Tensor& a, const Tensor& b) = delete; 
          maybestatic void methodname(const Tensor& out, Tensor&& a, const Tensor& b) = delete; 
          maybestatic void methodname(const Tensor& out, const Tensor& a, Tensor&& b) = delete; 
          maybestatic void methodname(Tensor&& out, Tensor&& a, const Tensor& b) = delete; 
          maybestatic void methodname(Tensor&& out, const Tensor& a, Tensor&& b) = delete; 
          maybestatic void methodname(const Tensor& out, Tensor&& a, Tensor&& b) = delete; 
          maybestatic void methodname(Tensor&& out, Tensor&& a, Tensor&& b) = delete;
        */
    }
}

#[macro_export] macro_rules! torch_disallow_temporaries {
    ($methodname:ident) => {
        /*
                TORCH_DISALLOW_TEMPORARIES_IMPL(methodname,)
        */
    };
    ($methodname:ident, static) => {
        /*
                TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, static)
        */
    }
}

impl TensorIteratorBase {

    pub fn build(&mut self, _0: &mut TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn foreach_reduced_elt(&mut self, 
        loop_:       LoopSubiter,
        parallelize: bool)  {
        let parallelize: bool = parallelize.unwrap_or(true);

        todo!();
        /*
        
        */
    }
    
    pub fn ndim(&self) -> i32 {
        
        todo!();
        /*
            return shape_.size();
        */
    }
    
    pub fn shape(&self) -> &[i32] {
        
        todo!();
        /*
            return shape_;
        */
    }
    
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn ntensors(&self) -> i32 {
        
        todo!();
        /*
            return operands_.size();
        */
    }
    
    pub fn noutputs(&self) -> i32 {
        
        todo!();
        /*
            return num_outputs_;
        */
    }
    
    pub fn ninputs(&self) -> i32 {
        
        todo!();
        /*
            return ntensors() - noutputs();
        */
    }
    
    pub fn view_offsets(&self) -> &[i32] {
        
        todo!();
        /*
            return view_offsets_;
        */
    }
    
    /**
      | number of elements in the output operand.
      | this is the same as numel() for operations
      | that are not reductions.
      |
      */
    pub fn num_output_elements(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }

    /// number of reduced dimensions in
    /// a reduction operation
    ///
    pub fn num_reduce_dims(&self) -> i32 {
        
        todo!();
        /*
        
        */
    }

    /// 1-dimensional iteration and no buffering
    /// or type conversion
    ///
    pub fn is_trivial_1d(&self) -> bool {
        
        todo!();
        /*
        
        */
    }

    /// Reducible to 1-dimensional and all
    /// operands are contiguous
    ///
    pub fn is_contiguous(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_dim_reduced(&self, dim: i32) -> bool {
        
        todo!();
        /*
        
        */
    }

    /// Accessors for each operand
    pub fn strides(&self, arg: i32) -> &[i32] {
        
        todo!();
        /*
            return operands_[arg].stride_bytes;
        */
    }
    
    pub fn data_ptr(&self, arg: i32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dtype(&self, arg: i32) -> ScalarType {
        let arg: i32 = arg.unwrap_or(0);

        todo!();
        /*
            return operands_[arg].current_dtype;
        */
    }
    
    pub fn common_dtype(&self) -> ScalarType {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined, "Queried for invalid common dtype!");
        return common_dtype_;
        */
    }
    
    pub fn input_dtype(&self, arg: i32) -> ScalarType {
        let arg: i32 = arg.unwrap_or(0);

        todo!();
        /*
            return operands_[num_outputs_ + arg].current_dtype;
        */
    }
    
    pub fn device(&self, arg: i32) -> Device {
        let arg: i32 = arg.unwrap_or(0);

        todo!();
        /*
            return operands_[arg].device;
        */
    }
    
    pub fn device_type(&self, arg: i32) -> DeviceType {
        let arg: i32 = arg.unwrap_or(0);

        todo!();
        /*
            return device(arg).type();
        */
    }
    
    pub fn element_size(&self, arg: i32) -> i64 {
        
        todo!();
        /*
            return elementSize(dtype(arg));
        */
    }
    
    pub fn is_scalar(&self, arg: i32) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_cpu_scalar(&self, arg: i32) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn tensor(&self, arg: i32) -> &Tensor {
        
        todo!();
        /*
            return *operands_[arg].tensor;
        */
    }
    
    pub fn output(&self, arg: i32) -> &Tensor {
        let arg: i32 = arg.unwrap_or(0);

        todo!();
        /*
            AT_ASSERT(arg < num_outputs_);
        return *operands_[arg].tensor;
        */
    }

    /**
      | Copies from temporary outputs back to the
      | original outputs
      |
      | NOTE: only used on CPU
      */
    pub fn cast_outputs(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn input(&self, arg: i32) -> Tensor {
        let arg: i32 = arg.unwrap_or(0);

        todo!();
        /*
            AT_ASSERT(arg >= 0 && arg < ntensors() - num_outputs_);
        return *operands_[num_outputs_ + arg].tensor;
        */
    }

    /// Removes an operand from this iterator
    ///
    pub fn remove_operand(&mut self, arg: i32)  {
        
        todo!();
        /*
        
        */
    }

    /// Shrinks an iterated dimension
    pub fn narrow(&mut self, 
        dim:   i32,
        start: i64,
        size:  i64)  {
        
        todo!();
        /*
        
        */
    }

    /// Narrows every dim after and including
    /// `start_dim` to size one.
    ///
    pub fn select_all_keeping_dim(&mut self, 
        start_dim: i32,
        starts:    &[i32])  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Replaces the data pointer for the operand at
      | index `arg`.
      |
      | The new pointer should have the same sizes,
      | strides and dtype as the original
      */
    pub fn unsafe_replace_operand(&mut self, 
        arg:  i32,
        data: *mut c_void)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Splits this TensorIterator into two
      | iterators. Together they iterate over the
      | entire operation. Used by
      | `with_32bit_indexing()`.
      |
      */
    pub fn split(&mut self, dim: i32) -> Box<TensorIterator> {
        
        todo!();
        /*
        
        */
    }

    /// Returns the dimension with the largest
    /// extent: (size[dim]-1) * stride[dim]
    ///
    pub fn get_dim_to_split(&self) -> i32 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn scalar_value<T>(&mut self, arg: i32) -> T {
    
        todo!();
        /*
            auto& op = operands_[arg];
        return fetch_and_cast<T>(op.tensor->scalar_type(), op.data);
        */
    }
    
    pub fn loop_2d_from_1d<loop1d_t>(&mut self, loop_: &Loop1d) -> Auto {
    
        todo!();
        /*
            return [loop, ntensor=ntensors()](
            char** base, const i64* strides, i64 size0, i64 size1) {
          PtrVector data(base, base + ntensor);
          const i64* outer_strides = &strides[ntensor];
          for (i64 i = 0; i < size1; i++) {
            if (i > 0) {
              for (i64 arg = 0; arg < ntensor; arg++) {
                data[arg] += outer_strides[arg];
              }
            }
            loop(data.data(), strides, size0);
          }
        };
        */
    }

    //template <typename loop1d_t, enable_if_t<is_convertible< loop1d_t, function_ref<void(char**, const i64* strides, i64 size)> >::value, int> = 0>
    pub fn for_each(&mut self, 
        loop_:      Loop1d,
        grain_size: i64)  {

        let grain_size: i64 = grain_size.unwrap_or(GRAIN_SIZE);

        todo!();
        /*
            for_each(loop_2d_from_1d(loop), grain_size);
        */
    }
    
    pub fn for_each(&mut self, 
        loop_:      Loop2d,
        grain_size: i64)  {
        let grain_size: i64 = grain_size.unwrap_or(GRAIN_SIZE);

        todo!();
        /*
        
        */
    }
    
    pub fn parallel_reduce(&mut self, loop_: Loop2d)  {
        
        todo!();
        /*
        
        */
    }

    //template <typename loop1d_t, enable_if_t<is_convertible< loop1d_t, function_ref<void(char**, const i64* strides, i64 size)> >::value, int> = 0>
    pub fn serial_for_each(&mut self, 
        loop_: Loop1d,
        range: Range)  {
        
        todo!();
        /*
            serial_for_each(loop_2d_from_1d(loop), range);
        */
    }
    
    pub fn serial_for_each(&self, 
        loop_: Loop2d,
        range: Range)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a strides array for a Tensor with
      | shape of this iterator. The parameter
      | `element_size` specifies the size of
      | Tensor's data type in bytes (e.g. `4` for
      | `float`)
      |
      */
    pub fn compatible_stride(&self, element_size: i32) -> StrideVector {
        
        todo!();
        /*
        
        */
    }

    /**
      | Inverts the re-ordering done by
      | reorder_dimensions. This can only be called
      | *before* coalesce_dimensions() is called.
      |
      */
    pub fn invert_perm(&self, input: &[i32]) -> DimVector {
        
        todo!();
        /*
        
        */
    }

    /**
      | Reapply same re-ordering as it is done by
      | reorder_dimensions. This can only be called
      | *before* coalesce_dimensions() is called.
      |
      */
    pub fn apply_perm_and_mul(&self, 
        input: &[i32],
        mul:   i32) -> DimVector {
        
        todo!();
        /*
        
        */
    }

    /// Helper functions for CPU iteration
    pub fn get_dim_strides(&self, dim: i32) -> StrideVector {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_strides(&self) -> StrideVector {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_inner_strides(&self) -> StrideVector {
        
        todo!();
        /*
            return get_dim_strides(0);
        */
    }
    
    pub fn get_base_ptrs(&self) -> PtrVector {
        
        todo!();
        /*
        
        */
    }

    /**
      | Helper functions for advanced stride
      | manipulations (e.g. torch.flip)
      |
      */
    pub fn unsafe_set_arg_strides(&mut self, 
        arg:     i32,
        strides: &[i32])  {
        
        todo!();
        /*
            operands_[arg].stride_bytes = move(strides);
        */
    }
    
    pub fn unsafe_set_arg_data(&mut self, 
        arg:  i32,
        data: *mut c_void)  {
        
        todo!();
        /*
            operands_[arg].data = data;
        */
    }

    /**
      | true if the stride computation can use
      | 32-bit arithmetic. Used by GPU kernels
      |
      */
    pub fn can_use_32bit_indexing(&self) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | An "iteratable" object that recursively
      | splits this iterator into sub-iterators that
      | can use 32-bit indexing.
      |
      */
    pub fn with_32bit_indexing(&self) -> SplitUntil32Bit {
        
        todo!();
        /*
        
        */
    }

    /**
      | If the kernel should accumulate into
      | the output. Only relevant for CUDA reductions.
      |
      */
    pub fn should_accumulate(&self) -> bool {
        
        todo!();
        /*
            return accumulate_;
        */
    }

    /**
      | Whether this iterator produces the actual
      | output, as opposed to something that will be
      | accumulated further. Only relevant for CUDA
      | reductions.
      |
      */
    pub fn is_final_output(&self) -> bool {
        
        todo!();
        /*
            return final_output_;
        */
    }
    
    pub fn has_contiguous_first_dim(&self) -> bool {
        
        todo!();
        /*
            int num_tensors = ntensors();
        for (int i = 0; i < num_tensors; i++) {
          if (strides(i)[0] != element_size(i)) {
            return false;
          }
        }
        return true;
        */
    }
    
    pub fn set_output(&mut self, 
        output_idx: i64,
        sizes:      &[i32],
        strides:    &[i32],
        options:    TensorOptions,
        names:      DimnameList)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn build_binary_float_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn build_borrowing_binary_float_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
        
        */
    }

    //TORCH_DISALLOW_TEMPORARIES(build_borrowing_binary_float_op)
    pub fn build_binary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn build_borrowing_binary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
        
        */
    }

    //TORCH_DISALLOW_TEMPORARIES(build_borrowing_binary_op)
    pub fn build_unary_float_op(&mut self, 
        out: &Tensor,
        a:   &Tensor)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn build_unary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Mutable reference as it moves tensors
      | out of TensorIteratorConfig
      |
      */
    pub fn populate_operands(&mut self, _0: &mut TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn mark_outputs(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn mark_resize_outputs(&mut self, _0: &TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_mem_overlaps(&mut self, _0: &TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_shape(&mut self, _0: &TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_strides(&mut self, _0: &TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn reorder_dimensions(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn permute_dimensions(&mut self, perm: &[i32])  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_types(&mut self, _0: &TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_common_dtype(&mut self) -> ScalarType {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allocate_or_resize_outputs(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn fast_set_up(&mut self, _0: &TensorIteratorConfig) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_fast_setup_type(&mut self, _0: &TensorIteratorConfig) -> FastSetupType {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_names(&mut self, _0: &TensorIteratorConfig)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn propagate_names_to_outputs(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn coalesce_dimensions(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct TensorIterator {
    base: TensorIteratorBase,
}

impl Default for TensorIterator {
    
    fn default() -> Self {
        todo!();
        /*
        : tensor_iterator_base(),

        
        */
    }
}

impl TensorIterator {

    /**
      | Slicing is OK, TensorIterator guaranteed
      | NOT to have any fields
      |
      */
    pub fn new(iter: &TensorIteratorBase) -> Self {
    
        todo!();
        /*
        : tensor_iterator_base(iter),

        
        */
    }
    
    pub fn binary_float_op(
        out: &mut Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn binary_op(
        out: &mut Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn borrowing_binary_op(
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }

    //TORCH_DISALLOW_TEMPORARIES(borrowing_binary_op)
    pub fn comparison_op(
        out: &mut Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unary_op(
        out: &mut Tensor,
        a:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unary_float_op(
        out: &mut Tensor,
        a:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn nullary_op(out: &mut Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn borrowing_nullary_op(out: &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn reduce_op(
        out: &mut Tensor,
        a:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn reduce_op(
        out1: &mut Tensor,
        out2: &mut Tensor,
        a:    &Tensor) -> TensorIterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn maybe_get_output(&mut self, output_idx: i64) -> &Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_output(&mut self, 
        output_idx: i64,
        sizes:      &[i32],
        strides:    &[i32],
        options:    TensorOptions,
        names:      DimnameList)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct TensorIteratorConfig {
    tensors:                         SmallVector<MaybeOwned<Tensor>,4>,
    num_outputs:                     i32, // default = 0
    num_inputs:                      i32, // default = 0
    static_shape:                    Option<DimVector>, // default = nullopt
    static_dtype_and_device:         Option<(ScalarType,Device)>, // default = nullopt
    check_mem_overlap:               bool, // default = true
    allow_cpu_scalars:               bool, // default = false
    is_reduction:                    bool, // default = false
    resize_outputs:                  bool, // default = true
    check_all_same_dtype:            bool, // default = true
    check_all_same_device:           bool, // default = true
    enforce_safe_casting_to_output:  bool, // default = false
    enforce_linear_iteration:        bool, // default = false
    promote_inputs_to_common_dtype:  bool, // default = false
    promote_integer_inputs_to_float: bool, // default = false
    cast_common_dtype_to_outputs:    bool, // default = false
}

impl TensorIteratorConfig {

    /**
      | Construction
      |
      | Stores input/output Tensors without
      | incrementing the reference count.
      |
      | Important: the outputs have to be added
      | before the inputs.
      |
      */
    pub fn add_output(&mut self, output: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            return add_borrowed_output(output);
        */
    }
    
    pub fn add_input(&mut self, input: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            return add_borrowed_input(input);
        */
    }

    /**
      | Stores input/output Tensors while
      | incrementing the reference count.
      |
      | Note that add_{in,out}put are nearly always
      | what you want, and the exception (adding an
      | unnamed temporary) won't compile.
      */
    pub fn add_owned_output(&mut self, output: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_owned_input(&mut self, input: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }

    /**
      | Advanced API: stores input/output Tensors
      |   without incrementing the reference count. The
      |   caller must ensure that these Tensors live at
      |   least as long as this TensorIteratorConfig
      |   and any
      |
      | TensorIteratorBase built from this
      | TensorIteratorConfig.
      |
      | Important: the outputs have to be added
      | before the inputs.
      |
      */
    pub fn add_borrowed_output(&mut self, output: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_borrowed_input(&mut self, input: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }

    /**
      | Sets the check_mem_overlap_ flag, which is true
      |  by default.
      |
      | If true, inputs are checked for partial
      | overlap with the outputs and outputs are
      | checked for internal overlap
      | (e.g. broadcasted views). An error is raised
      | if unacceptable overlap is detected.
      |
      | If you're migrating an existing operator to
      | using TensorIterator, please consider if the
      | previous implementation checked memory
      | overlap. If it did not, and if the operator
      | is idempotent (for example, Tensor.fill_(0)),
      | then checking memory overlap is
      | BC-breaking. Please don't check memory
      | overlap in that case.
      |
      */
    pub fn set_check_mem_overlap(&mut self, check_mem_overlap: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            check_mem_overlap_ = check_mem_overlap;
        return *this;
        */
    }

    /**
      | Sets the check_all_same_dtype_ flag, which is
      | true by default
      |
      | If true, checks that all inputs and defined
      | outputs have the same dtype
      |
      | Setting either of
      |   promote_inputs_to_common_dtype_ or
      |   cast_common_dtype_to_outputs_ to true will
      |   set check_all_same_dtype_ to false.
      */
    pub fn check_all_same_dtype(&mut self, check_all_same_dtype: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            check_all_same_dtype_ = _check_all_same_dtype;
        return *this;
        */
    }

    /**
      | Sets the check_all_same_device_ flag, which
      | is true by default
      |
      | If true, all operands must be on the same
      |   device, with the possible exception of CPU
      |   scalars, which can be passed to some CUDA
      |   kernels as kernel arguments.
      */
    pub fn check_all_same_device(&mut self, check_all_same_device: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            check_all_same_device_ = _check_all_same_device;
        return *this;
        */
    }

    /**
      | Sets the enforce_safe_casting_to_output_
      | flag, which is false by default
      |
      | If true, the iterator's "common dtype" must
      |   be computable (see the [Common Dtype
      |   Computation] note) and canCast(common
      |   dtype, output dtype) must be true for all
      |   outputs.
      |
      */
    pub fn enforce_safe_casting_to_output(&mut self, enforce_safe_casting_to_output: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            enforce_safe_casting_to_output_ = _enforce_safe_casting_to_output;
        return *this;
        */
    }

    /**
      | Sets the enforce_linear_iteration_ flag,
      | which is false by default.
      |
      | If true, iteration goes in the same order as
      | a C-contiguous tensor is layed out in
      | memory. i.e. last dimension iterates fastest.
      |
      | This iteration order can be less efficient
      | and may even prevent vectorization.
      |
      | So only use if the correctness of your kernel
      | depends on it.
      |
      */
    pub fn enforce_linear_iteration(&mut self, enforce_linear_iteration: bool) -> &mut TensorIteratorConfig {

        let enforce_linear_iteration: bool = enforce_linear_iteration.unwrap_or(true);

        todo!();
        /*
            enforce_linear_iteration_ = _enforce_linear_iteration;
        return *this;
        */
    }

    /**
      | Sets the promote_inputs_to_common_dtype_
      | flag, which is false by default
      |
      | If true, the iterator's "common dtype" is
      |   always computed (see the [Common Dtype
      |   Computation] note) and, on the CPU,
      |   temporary copies of the inputs in the
      |   common dtype are passed as the actual
      |   inputs to the operation.
      |
      | Setting this flag to true sets
      | check_all_same_dtype_ to false.
      |
      */
    pub fn promote_inputs_to_common_dtype(&mut self, promote_inputs_to_common_dtype: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            promote_inputs_to_common_dtype_ = _promote_inputs_to_common_dtype;
        if (_promote_inputs_to_common_dtype) {
          check_all_same_dtype_ = false;
        }
        return *this;
        */
    }

    /**
      | Sets the promote_integer_inputs_to_float_
      | flag, which is false by default
      |
      | NOTE: If set to true, the
      | promote_inputs_to_common_dtype_ must also be
      | true.
      |
      | If true, if the iterator's "common dtype" is
      |   an integral type (including bool) then it
      |   is changed to the default float scalar
      |   type.
      */
    pub fn promote_integer_inputs_to_float(&mut self, promote_integer_inputs_to_float: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            promote_integer_inputs_to_float_ = _promote_integer_inputs_to_float;
        TORCH_INTERNAL_ASSERT(!promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_);
        return *this;
        */
    }
    
    pub fn is_reduction(&mut self, is_reduction: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            is_reduction_ = _is_reduction;
        return *this;
        */
    }
    
    pub fn allow_cpu_scalars(&mut self, allow_cpu_scalars: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            allow_cpu_scalars_ = _allow_cpu_scalars;
        return *this;
        */
    }

    /**
      | Sets the cast_common_dtype_to_outputs_ flag,
      | which is false by default
      |
      | If true, the iterator's "common dtype" must
      |   be computatable (see the [Common Dtype
      |   Computation] note) and, on the CPU,
      |   temporary copies of the outputs are passed
      |   as the actual output to the operation.
      |
      |   These temporaries are then copied to the
      |   original outputs after the operation is
      |   performed (see cast_outputs()).
      |
      | Setting this flag to true sets
      | check_all_same_dtype_ to false.
      |
      */
    pub fn cast_common_dtype_to_outputs(&mut self, cast_common_dtype_to_outputs: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            cast_common_dtype_to_outputs_ = _cast_common_dtype_to_outputs;
        if (_cast_common_dtype_to_outputs) {
          check_all_same_dtype_ = false;
        }
        return *this;
        */
    }
    
    pub fn resize_outputs(&mut self, resize_outputs: bool) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            resize_outputs_ = resize_outputs;
        return *this;
        */
    }

    /**
      | Bypass output dtype/device computation
      | and fix the dtype/device as specified
      | here.
      |
      */
    pub fn declare_static_dtype_and_device(&mut self, 
        dtype:  ScalarType,
        device: Device) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }
    
    pub fn declare_static_shape(&mut self, shape: &[i32]) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }
    
    pub fn declare_static_shape(&mut self, 
        shape:       &[i32],
        squash_dims: &[i32]) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
        
        */
    }

    /**
      | It would be better if this was && qualified,
      | but this would be at the cost of a lot of
      | boilerplate above
      |
      */
    pub fn build(&mut self) -> TensorIterator {
        
        todo!();
        /*
            TensorIterator iter;
        iter.build(*this);
        return iter;
        */
    }
}

pub struct SplitUntil32BitIterator {

    /**
      | stack of TensorIterators to be split
      |
      */
    vec: Vec<Box<TensorIterator>>,
}

impl PartialEq<SplitUntil32BitIterator> for SplitUntil32BitIterator  {
    
    #[inline] fn eq(&self, other: &Iterator) -> bool {
        todo!();
        /*
            // two iterators are equal if they are the same object or they're both empty
          return this == &other || (vec.empty() && other.vec.empty());
        */
    }
}

impl SplitUntil32BitIterator {
    
    pub fn new(iter: &TensorIteratorBase) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn prefix_increment(&mut self) -> &mut SplitUntil32BitIterator {
        
        todo!();
        /*
        
        */
    }
}

/**
  | A container-like struct that acts as if it
  | contains splits of a
  |
  | TensorIterator that can use 32-bit
  | indexing. Taken together the splits cover the
  | original TensorIterator.
  |
  */
pub struct SplitUntil32Bit {
    iter: &TensorIteratorBase,
}

impl SplitUntil32Bit {
    
    pub fn new(iter: &TensorIteratorBase) -> Self {
    
        todo!();
        /*
        : iter(iter),

        
        */
    }
    
    pub fn begin(&self) -> Iterator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn end(&self) -> Iterator {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorIterator.cpp]

pub type DimMask      = TensorIteratorBaseDimMask;
pub type PtrVector    = TensorIteratorBasePtrVector;
pub type Loop2d       = TensorIteratorBaseLoop2d;
pub type StrideVector = TensorIteratorBaseStrideVector;

#[inline] pub fn get_base_ptrs(
        ptrs:     *mut *mut u8,
        operands: &[OperandInfo])  {
    
    todo!();
        /*
            transform(operands.begin(), operands.end(), ptrs, [](const OperandInfo& op) {
        return static_cast<char*>(op.data);
      });
        */
}

#[inline] pub fn get_strides(
    strides:  *mut i64,
    operands: &[OperandInfo],
    ndim:     i64)  {
    
    todo!();
        /*
            for (i64 dim = 0; dim < ndim; ++dim) {
        for (usize arg = 0; arg < operands.size(); ++arg) {
          *strides++ = operands[arg].stride_bytes[dim];
        }
      }
      // Always at least 2d strides to support 2d for_each loops
      if (ndim < 2) {
        const i64 ntensors = operands.size();
        fill_n(strides, (2 - ndim) * ntensors, 0);
      }
        */
}

impl TensorIteratorConfig {
    
    pub fn add_owned_output(&mut self, output: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          num_inputs_ == 0,
          "Keep in mind that you have to add all outputs first before adding any input. "
          "For more details, see https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator.");
      tensors_.push_back(MaybeOwned<Tensor>::owned(in_place, output));
      num_outputs_++;
      return *this;
        */
    }
    
    pub fn add_owned_input(&mut self, input: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            tensors_.push_back(MaybeOwned<Tensor>::owned(in_place, input));
      num_inputs_++;
      return *this;
        */
    }
    
    pub fn add_borrowed_output(&mut self, output: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          num_inputs_ == 0,
          "Keep in mind that you have to add all outputs first before adding any input. "
          "For more details, see https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator.");
      tensors_.push_back(MaybeOwned<Tensor>::borrowed(output));
      num_outputs_++;
      return *this;
        */
    }
    
    pub fn add_borrowed_input(&mut self, input: &Tensor) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            tensors_.push_back(MaybeOwned<Tensor>::borrowed(input));
      num_inputs_++;
      return *this;
        */
    }
    
    pub fn declare_static_dtype_and_device(&mut self, 
        dtype:  ScalarType,
        device: Device) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            TORCH_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
      static_dtype_and_device_ = make_optional(make_pair(dtype, device));
      return *this;
        */
    }
    
    pub fn declare_static_shape(&mut self, shape: &[i32]) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            // WARNING:
      //   This will bypass all shape checking in the TensorIterator. Kernels which call this method
      //   are expected to check shapes before calling `add_owned_input` or `add_owned_output`.
      TORCH_CHECK(!resize_outputs_, "resize_outputs() must be called before declare_static_shape(...)")
      static_shape_ = make_optional(DimVector(shape));
      return *this;
        */
    }
    
    pub fn declare_static_shape(&mut self, 
        shape:       &[i32],
        squash_dims: &[i32]) -> &mut TensorIteratorConfig {
        
        todo!();
        /*
            declare_static_shape(shape);
      if (!static_shape_->size()) return *this;
      for (const auto& squash_dim : squash_dims) {
        TORCH_CHECK(squash_dim >= 0 && squash_dim < static_cast<i64>(static_shape_->size()),
                    "squash_dim ", squash_dim, " must be in [0, ", static_shape_->size(), ").");
        (*static_shape_)[squash_dim] = 1;
      }
      return *this;
        */
    }
}

impl TensorIteratorBase {

     /**
      | NOTE: [Computing output strides]
      |
      | We use the following algorithm to compute
      | output strides
      |
      | If correctly sized output is provided, we
      | respect its stides and don't change them
      |
      | Otherwise, if provided output is of incorrect
      | size or no output is provided, we try to
      | recover permutation that was applied to the
      | inputs by sorting the strides of the
      | inputs. Precedence is given to the inputs in
      | the order they were added, and to permutations
      | involving non-broadcasted dimensions
      |
      | 1. we loop over inputs starting from the first
      |
      | 2. for all inputs strides of broadcasted
      | dimensions are set to 0, and 0 compares equal
      | to anything. If one of the dimensions being
      | compared has a stride of 0, we move on to the
      | next tensor to determine if these dimensions
      | need to be swapped.
      |
      | 3. strides of dimensions equal to 1 participate
      | in sorting
      |
      | 4. if 2 strides are equal and neither is 0, we
      | try to break the tie by looking at the
      | corresponding dimensions of the
      | tensor. Dimensions were permuted if, when
      | iterating from the end, dimensions
      | corresponding to the same strides are
      | increasing. If dimensions are non-increasing,
      | we move on to the next input to break the tie.
      |
      | Instead of applying rule 4 for tie breaking, we
      | could move on to the next tensor directly. This
      | would result in possibly losing the correct
      | permuation of the first tensor if there are
      | permuted trivial dimensions, but could
      | potentially improve traversal order of the
      | second tensor. We chose the former option to
      | better propagate channels last layout for
      | example for a tensor with the sizes N1H1
      |
      | These rules result in the intuitive behavior
      | that in most cases recovers permutation of
      | either the first argument (if all arguments are
      | of the same size) or the argument that is not
      | broadcasted, regardless of its position.
      |
      | As a bonus, it also result in reasonably
      | well-behaved traversal order of the inputs and
      | outputs - in the kernels output is traversed
      | linearly, and since it closely follows input
      | layouts, inputs are traversed linearly as well
      |
      | Examples:
      |
      | full size tensor + broadcasted tensor with 0 or
      | 1 non-trivial dimensions => strides of output
      | are same as strides of full size input
      | regardless of the order
      |
      | 2 tensors of same size but different strides =>
      | output strides are the same as first argument
      |
      | We also have fast path for memory-dense inputs
      | with the same strides (or, trivially, single
      | memory-dense input) that outputs a tensor with
      | the same strides as inputs. The only difference
      | in result with the algorithm described above is
      | for strides for trivial (1) dimensions, where
      | in ambiguous cases for performance reasons we
      | default to contiguous strides.
      |
      | Example: tensor with sizes NC11 and strides
      | C1CC will produce output with strides C111
      | (note differences are only in the strides of
      | trivial dimensions, so physical layout is
      | unaffected but permutation information is lost)
      |
      | We might change this behavior in future once
      | performance considerations are resolved
      */
    pub fn reorder_dimensions(&mut self)  {
        
        todo!();
        /*
            // Sort the dimensions based on strides in ascending order with reduced dims
      // at the front. NOTE: that this inverts the order of C-contiguous tensors.
      // strides[0] is the fastest moving dimension instead of strides[ndim - 1].
      // See NOTE: [Computing output strides] and inline  comments for more detailed description

      perm_.resize(ndim());
      if (ndim() == 1) {
        perm_[0] = 0;
        return;
      }

      // initialize perm with n-1, n-2, ..., 1, 0
      iota(perm_.rbegin(), perm_.rend(), 0);

      // Reordering dimensions changes iteraton order
      if (enforce_linear_iteration_) {
        permute_dimensions(perm_);
        return;
      }

      // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
      // before dim1, and 0 if the comparison is ambiguous.
      auto should_swap = [&](usize dim0, usize dim1) {
        for (int arg = 0; arg < ntensors(); arg++) {
          // ignore undefined or incorrectly sized tensors
          if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
            continue;
          }
          i64 stride0 = operands_[arg].stride_bytes[dim0];
          i64 stride1 = operands_[arg].stride_bytes[dim1];
          if (is_reduction_ && operands_[arg].is_output) {
            // move reduced dimensions to the front
            // strides of reduced dimensions are always set to 0 by review_reduce_result
            if ((stride0 == 0) != (stride1 == 0)) {
              return stride1 == 0 ? 1 : -1;
            }
          }
          //move on to the next input if one of the dimensions is broadcasted
          if (stride0 == 0 || stride1 == 0) {
            continue;
          // it is important to return here only with strict comparisons, for equal strides we try to break the tie later
          // by comparing corresponding dimensions or if that does not work, moving on to the next tensor
          } else if (stride0 < stride1) {
            return -1;
          } else  if (stride0 > stride1) {
            return 1;
          } else { //equal strides, use dimensions themselves as the tie-breaker.
            //at this point, with zero strides out of the way, we are guaranteed that operand dimensions are equal to shape_
             auto t_dim0 = shape_[dim0];
             auto t_dim1 = shape_[dim1];
             //return only if dimensions should be swapped, otherwise move on to the next tensor
             if (t_dim0 > t_dim1) {
                 return 1;
             }
          }
        }
        return 0;
      };

      // insertion sort with support for ambiguous comparisons
      for (int i = 1; i < ndim(); i++) {
        int dim1 = i;
        for (int dim0 = i - 1; dim0 >= 0; dim0--) {
          int comparison = should_swap(perm_[dim0], perm_[dim1]);
          if (comparison > 0) {
            swap(perm_[dim0], perm_[dim1]);
            dim1 = dim0;
          } else if (comparison < 0) {
            break;
          }
        }
      }

      // perform re-ordering of shape and strides
      permute_dimensions(perm_);
        */
    }

    /**
      | Computes a common dtype using type promotion
      | 
      | See the [Common Dtype Computation]
      | note
      |
      */
    pub fn compute_common_dtype(&mut self) -> ScalarType {
        
        todo!();
        /*
            native::ResultTypeState state = {};
      for (const auto& op : operands_) {
        if (op.is_output) {
          continue;
        }

        state = native::update_result_type_state(*op.tensor, state);
      }

      common_dtype_ = native::result_type(state);
      TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);

      return common_dtype_;
        */
    }
}

pub fn original_options(op: &OperandInfo) -> TensorOptions {
    
    todo!();
        /*
            if (op.original_tensor->defined()) {
        return op.original_tensor->options();
      } else {
        return op.options();
      }
        */
}

#[macro_export] macro_rules! binary_float_op_config {
    () => {
        /*
        
          TensorIteratorConfig()                        
            .set_check_mem_overlap(true)                
            .allow_cpu_scalars(true)                    
            .promote_inputs_to_common_dtype(true)       
            .cast_common_dtype_to_outputs(true)         
            .enforce_safe_casting_to_output(true)       
            .promote_integer_inputs_to_float(true)
        */
    }
}

// This cannot be a function because TensorIteratorConfig is not
// copyable or movable, so it can't be returned from the function.
#[macro_export] macro_rules! binary_op_config {
    () => {
        /*
        
          TensorIteratorConfig()                                
            .set_check_mem_overlap(true)                        
            .allow_cpu_scalars(true)                            
            .promote_inputs_to_common_dtype(true)               
            .cast_common_dtype_to_outputs(true)                 
            .enforce_safe_casting_to_output(true)               
        */
    }
}

impl TensorIteratorBase {
    
    /**
      | Implements the the behavior of the following
      | flags:
      |
      |   - check_all_same_dtype_
      |   - check_all_same_device_
      |   - enforce_safe_casting_to_output_
      |   - promote_inputs_to_common_dtype_
      |   - cast_common_dtype_to_outputs_
      |
      | See their descriptions in TensorIterator.h for
      | details.
      |
      | NOTE: Checks for more specific behaviors
      |   (e.g. the first and second inputs must share
      |   a dtype, but the third must have the long
      |   dtype) should be implemented directly and
      |   outside of TensorIterator.
      |
      */
    pub fn compute_types(&mut self, config: &TensorIteratorConfig)  {
        
        todo!();
        /*
            // Reviews operands (1/2)
      //   - validates that all input tensors are defined
      //   - computes common device
      //   - determines if there are undefined outputs
      //   - determines if there are different dtypes and attempts
      //       to quickly acquire a common dtype
      Device common_device = kCPU;
      common_dtype_ = ScalarType::Undefined;
      // NB: despite output_dtype's generic sounding name, it only is
      // used in a nontrivial way if check_all_same_dtype is true
      ScalarType output_dtype = ScalarType::Undefined;
      bool has_different_input_dtypes = false;
      bool has_different_output_dtypes = false;
      bool has_undefined_outputs = false;

      for (auto& op : operands_) {
        // Validates that all inputs have type information, and that
        //   if an output is missing type information that we can infer
        //   the device it should be allocated on.
        if (!op.is_type_defined()) {
          TORCH_INTERNAL_ASSERT(op.is_output, "Found type undefined input tensor!");
          if (config.static_dtype_and_device_.has_value()) {
            op.target_dtype = config.static_dtype_and_device_->first;
            op.device = config.static_dtype_and_device_->second;
          } else {
            TORCH_INTERNAL_ASSERT(config.check_all_same_device_);
            has_undefined_outputs = true;
            continue;
          }
        }

        // Validates input tensors are defined
        if (!op.tensor->defined()) {
          TORCH_INTERNAL_ASSERT(op.is_output, "Found undefined input tensor!");
          continue;
        }

        TORCH_INTERNAL_ASSERT(op.target_dtype == op.current_dtype)

        // Acquires the first non-CPU device (if any) as the common device
        if (common_device == kCPU && !op.tensor->is_cpu()) {
          common_device = op.tensor->device();
        }

        if (!op.is_output) {
          // Determines if there are varying input dtypes
          // NOTE: the common dtype is set to the first defined input dtype observed
          if (op.target_dtype != common_dtype_) {
            if (common_dtype_ == ScalarType::Undefined) {
              common_dtype_ = op.target_dtype;
            } else {
              has_different_input_dtypes = true;
            }
          }
        } else {  // op.is_output
          // Determines if there are varying output dtypes
          // NOTE: the output dtype is set to the first defined output dtype observed
          if (op.target_dtype != output_dtype) {
            if (output_dtype == ScalarType::Undefined) {
              output_dtype = op.target_dtype;
            } else {
              has_different_output_dtypes = true;
            }
          }
        }
      }

      // Checks that either the computation type is computable or unneeded
      TORCH_INTERNAL_ASSERT(!(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
                            (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
                            config.cast_common_dtype_to_outputs_)));

      // Checks that all inputs and defined outputs are the same dtype, if requested
      if (config.check_all_same_dtype_ &&
          (has_different_input_dtypes || has_different_output_dtypes ||
          (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
        // Throws an informative error message
        for (auto& op : operands_) {
          if (!op.tensor->defined()) {
            continue;
          }

          TORCH_CHECK(op.target_dtype == common_dtype_,
                      "Found dtype ", op.target_dtype, " but expected ", common_dtype_);
        }
      }

      // Short-circuits if no additional work required
      if (!has_undefined_outputs && !config.check_all_same_device_ &&
          !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
          !config.enforce_safe_casting_to_output_) {
        // Invalidates common_dtype_ if it could not be inferred
        common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
        return;
      }

      // Computes a common dtype, if needed
      if (has_different_input_dtypes && config.promote_inputs_to_common_dtype_) {
        common_dtype_ = compute_common_dtype();
      }

      // Promotes common dtype to the default float scalar type, if needed
      if (config.promote_integer_inputs_to_float_ &&
          isIntegralType(common_dtype_, /*includeBool=*/true)) {
        common_dtype_ = typeMetaToScalarType(get_default_dtype());
      }

      // Reviews operands (2/2)
      //   - sets metadata for undefined outputs
      //   - checks that all tensors are on the same device, if requested
      //   - checks that the common dtype can safely cast to each output, if requested
      //   - creates temporaries for CPU operations, if needed and requested
      int max_cpu_scalars_on_non_cpu = config.allow_cpu_scalars_ ? 1 : 0;
      int current_cpu_scalars_on_non_cpu = 0;
      for (auto& op : operands_) {
        if (!op.is_type_defined()) {
          op.target_dtype = common_dtype_;
          op.device = common_device;
          continue;
        }

        // Skips undefined tensors
        if (!op.tensor->defined()) {
          continue;
        }

        // Checks all tensors are on the same device, if requested
        if (config.check_all_same_device_) {
          // Handles CPU scalars on CUDA kernels that support them
          if (!common_device.is_cpu() &&
              config.allow_cpu_scalars_ && !op.is_output && op.tensor->dim() == 0 &&
              op.tensor->is_cpu()) {
            TORCH_CHECK(current_cpu_scalars_on_non_cpu < max_cpu_scalars_on_non_cpu,
                        "Trying to pass too many CPU scalars to non-CPU kernel!");
            ++current_cpu_scalars_on_non_cpu;
          } else if (op.device != common_device) {
            TORCH_CHECK(false,
                        "Expected all tensors to be on the same device, but "
                        "found at least two devices, ", common_device, " and ", op.device, "!");
          }
        }

        // Checks safe casting, if requested
        if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
          TORCH_CHECK(canCast(common_dtype_, op.current_dtype),
                      "result type ", common_dtype_, " can't be cast to the "
                      "desired output type ", op.current_dtype);
        }

        // Creates temporaries for CPU operations, if needed and requested
        // TODO: reuse temporaries when possible (e.g. for inplace operations)
        if (common_device == kCPU) {
          // Casts to outputs by creating temporaries of the correct dtype (if needed)
          // NB: we skip this on is_meta_, because the temporary allocation here is
          // unnecessary if we aren't going to actually do the compute
          if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_ && !is_meta_) {
            TORCH_INTERNAL_ASSERT(op.tensor->defined());
            // Marker [Output original_tensor is set]
            op.original_tensor = op.tensor;
            // NB: do NOT use set_output here, as the temporary is NOT a true output;
            // op.tensor is the true output and it was pre-provided for us.
            // TODO: The logic for cast_outputs will need to be handled by the
            // structured kernels implementation.  What probably should happen
            // is that we pass in the inferred dtype into the out kernel, and
            // then after calling the out kernel, do the conversion (which
            // is cast_outputs here), but integrating this with existing
            // TensorIterator will take a little doing
            op.tensor = MaybeOwned<Tensor>::owned(
                empty_like(*op.tensor,
                               op.tensor->options().dtype(common_dtype_),
                               LEGACY_CONTIGUOUS_MEMORY_FORMAT));
            if (!names_.empty()) {
              namedinference::propagate_names(*op.tensor, names_);
            }
            op.current_dtype = common_dtype_;
            op.target_dtype = common_dtype_;
          }

          // Promotes inputs by creating temporaries of the correct dtype
          if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
            op.original_tensor = op.tensor;
            op.tensor = MaybeOwned<Tensor>::owned(op.tensor->to(common_dtype_));
            op.current_dtype = common_dtype_;
            op.target_dtype = common_dtype_;
          }
        }
        common_device_ = common_device;
      }
        */
    }
    
    pub fn compatible_stride(&self, element_size: i32) -> StrideVector {
        
        todo!();
        /*
            auto stride = StrideVector();
      i64 next_stride = element_size;
      for (int dim = 0; dim < ndim(); dim++) {
        stride.push_back(next_stride);
        next_stride *= shape_[dim];
      }
      return stride;
        */
    }
    
    pub fn invert_perm(&self, input: &[i32]) -> DimVector {
        
        todo!();
        /*
            // Invert the permutation caused by reorder_dimensions. This is not valid
      // after coalesce_dimensions is called.
      TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);
      TORCH_INTERNAL_ASSERT(input.size()==perm_.size());
      auto res = DimVector(input.size()); //no initialization needed, every value in res should be written to.
      for (int dim = 0; dim < ndim(); dim++) {
        res[perm_[dim]] = input[dim];
      }
      return res;
        */
    }
    
    pub fn allocate_or_resize_outputs(&mut self)  {
        
        todo!();
        /*
            for (int i = 0; i < num_outputs_; i++) {
        auto& op = operands_[i];
        if (!op.tensor->defined() || op.will_resize) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
          int element_size = elementSize(op.target_dtype);
          op.stride_bytes = compatible_stride(element_size);
          // check if permutation is just an inverted order
          bool inverted = true;
          for (int i = 0; i < ndim(); i++) {
            if (perm_[i] != ndim() - i - 1) {
              inverted = false;
              break;
            }
          }
          auto tensor_shape = invert_perm(shape_);
          if (inverted) {
            // can just return contiguous output
            // it is faster because it avoids allocating 0 size tensor and
            // resizing and restriding it
            set_output(i, tensor_shape, {}, original_options(op), names_);
          } else {
            auto tensor_stride = invert_perm(op.stride_bytes);
            for (int dim = 0; dim < ndim(); dim++) {
              tensor_stride[dim] /= element_size;
            }
            set_output(i, tensor_shape, tensor_stride, original_options(op), names_);
          }
          op.current_dtype = op.target_dtype;
        } else if (op.tensor->defined()) {
          // Even if we don't resize, we still need to tell set_output about
          // the output, so that we properly set guard and propagate names
          set_output(i, op.tensor->sizes(), {}, original_options(op), names_);
        }
      }
        */
    }
    
    pub fn compute_names(&mut self, config: &TensorIteratorConfig)  {
        
        todo!();
        /*
            bool should_infer_names = any_of(
          operands_.begin(),
          operands_.end(),
          [](const OperandInfo& op) {
            return op.tensor->defined() && op.tensor->has_names();
          });
      if (!should_infer_names) {
        return;
      }

      for (auto& op : operands_) {
        if (!op.tensor->defined()) continue;
        // Don't include output tensors if we are resizing, since we will
        // clobber their names in any case.  (If the output tensor was
        // also an input tensor, we'll pick it up when it shows up again
        // in operands).
        if (config.resize_outputs_ && op.is_output) continue;
        // perform name inference
        if (names_.empty()) {
          names_ = op.tensor->names();
        } else {
          names_ = NameVector(unify_from_right(names_, op.tensor->names()));
        }
      }
        */
    }
    
    pub fn coalesce_dimensions(&mut self)  {
        
        todo!();
        /*
            if (ndim() <= 1) {
        return;
      }

      // We can coalesce two adjacent dimensions if either dim has size 1 or if:
      // shape[n] * stride[n] == shape[n + 1].
      auto can_coalesce = [&](int dim0, int dim1) {
        auto shape0 = shape_[dim0];
        auto shape1 = shape_[dim1];
        if (shape0 == 1 || shape1 == 1) {
          return true;
        }
        for (int i = 0; i < ntensors(); i++) {
          auto& stride = operands_[i].stride_bytes;
          if (shape0 * stride[dim0] != stride[dim1]) {
            return false;
          }
        }
        return true;
      };

      // replace each operands stride at dim0 with its stride at dim1
      auto replace_stride = [&](int dim0, int dim1) {
        for (int i = 0; i < ntensors(); i++) {
          auto& stride = operands_[i].stride_bytes;
          stride[dim0] = stride[dim1];
        }
      };

      int prev_dim = 0;
      for (int dim = 1; dim < ndim(); dim++) {
        if (can_coalesce(prev_dim, dim)) {
          if (shape_[prev_dim] == 1) {
            replace_stride(prev_dim, dim);
          }
          shape_[prev_dim] *= shape_[dim];
        } else {
          prev_dim++;
          if (prev_dim != dim) {
            replace_stride(prev_dim, dim);
            shape_[prev_dim] = shape_[dim];
          }
        }
      }

      shape_.resize(prev_dim + 1);
      for (int i = 0; i < ntensors(); i++) {
        operands_[i].stride_bytes.resize(ndim());
      }
      has_coalesced_dimensions_ = true;
        */
    }
    
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
            i64 numel = 1;
      for (i64 size : shape_) {
        numel *= size;
      }
      return numel;
        */
    }
    
    pub fn get_dim_strides(&self, dim: i32) -> StrideVector {
        
        todo!();
        /*
            auto dims = ndim();
      auto inner_strides = StrideVector();
      for (auto& op : operands_) {
        inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
      }
      return inner_strides;
        */
    }
    
    pub fn get_base_ptrs(&self) -> SmallVector<*mut u8,4> {
        
        todo!();
        /*
            auto ptrs = SmallVector<char*, 4>(ntensors());
      get_base_ptrs(ptrs.data(), operands_);
      return ptrs;
        */
    }
    
    pub fn is_dim_reduced(&self, dim: i32) -> bool {
        
        todo!();
        /*
            for (auto& op : operands_) {
        if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
          return true;
        }
      }
      return false;
        */
    }
    
    pub fn permute_dimensions(&mut self, perm: &[i32])  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(perm.size() == static_cast<unsigned>(ndim()));

      auto reorder = [perm](IntArrayRef data) {
        auto res = DimVector(data.size(), 0);
        for (usize i = 0; i < perm.size(); i++) {
          res[i] = data[perm[i]];
        }
        return res;
      };

      // Update shape and strides
      shape_ = reorder(shape_);
      for (auto& op : operands_) {
        if (op.stride_bytes.size() > 0) {
          op.stride_bytes = reorder(op.stride_bytes);
        }
      }
        */
    }
    
    pub fn num_output_elements(&self) -> i64 {
        
        todo!();
        /*
            i64 elem = 1;
      for (int dim = 0; dim < ndim(); dim++) {
        if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0)  {
          elem *= shape_[dim];
        }
      }
      return elem;
        */
    }
    
    pub fn num_reduce_dims(&self) -> i32 {
        
        todo!();
        /*
            int count = 0;
      for (int dim = 0; dim < ndim(); dim++) {
        if (operands_[0].stride_bytes[dim] == 0) {
          count++;
        }
      }
      return count;
        */
    }
    
    pub fn for_each(&mut self, 
        loop_:      Loop2d,
        grain_size: i64)  {
        
        todo!();
        /*
            i64 numel = this->numel();
      if (numel == 0) {
        return;
      } else if (numel < grain_size || get_num_threads() == 1) {
        return serial_for_each(loop, {0, numel});
      } else {
        parallel_for(0, numel, grain_size, [&](i64 begin, i64 end) {
          serial_for_each(loop, {begin, end});
        });
      }
        */
    }
    
    pub fn get_strides(&self) -> StrideVector {
        
        todo!();
        /*
            const auto dim = ndim();
      StrideVector strides(max(dim, 2) * ntensors());
      get_strides(strides.data(), operands_, dim);
      return strides;
        */
    }
    
    pub fn serial_for_each(&self, 
        loop_: Loop2d,
        range: Range)  {
        
        todo!();
        /*
            if (range.size() == 0) {
        return;
      }

      const auto ntensors = this->ntensors();
      const auto ndim = this->ndim();

      SmallBuffer<char*, 4> ptrs(ntensors);
      SmallBuffer<i64, 8> strides(ntensors * max(ndim, 2));

      get_base_ptrs(ptrs.data(), operands_);
      get_strides(strides.data(), operands_, ndim);
      internal::serial_for_each(
          shape_, strides, ptrs.data(), ptrs.size(), loop, range);
        */
    }
    
    pub fn is_trivial_1d(&self) -> bool {
        
        todo!();
        /*
            // TODO: check for casting once it's supported
      return ndim() == 1;
        */
    }
    
    pub fn is_contiguous(&self) -> bool {
        
        todo!();
        /*
            if (numel() == 1) {
        return true;
      }
      if (ndim() != 1) {
        return false;
      }
      return has_contiguous_first_dim();
        */
    }
    
    pub fn is_scalar(&self, arg: i32) -> bool {
        
        todo!();
        /*
            const auto& stride = operands_[arg].stride_bytes;
      for (int i = 0; i < ndim(); i++) {
        if (stride[i] != 0 && shape_[i] != 1) {
          return false;
        }
      }
      return true;
        */
    }
    
    pub fn is_cpu_scalar(&self, arg: i32) -> bool {
        
        todo!();
        /*
            return is_scalar(arg) && device(arg).is_cpu();
        */
    }
    
    pub fn cast_outputs(&mut self)  {
        
        todo!();
        /*
            for (auto& op : operands_) {
        if (op.is_output && op.original_tensor->defined() &&
            op.original_tensor->scalar_type() != op.current_dtype) {
          // TODO: Now that set_output resizes both the original_tensor
          // and tensor, this condition should no longer ever be true
          if (op.original_tensor->sizes() != op.tensor->sizes()){
            op.original_tensor->resize_as_(*op.tensor).as_strided_(op.tensor->sizes(), op.tensor->strides());
          }
          op.original_tensor->copy_(*op.tensor);
          op.tensor = op.original_tensor;
        }
      }
        */
    }
    
    pub fn data_ptr(&self, arg: i32)  {
        
        todo!();
        /*
            return operands_[arg].data;
        */
    }
    
    pub fn remove_operand(&mut self, arg: i32)  {
        
        todo!();
        /*
            operands_.erase(operands_.begin() + arg);
        */
    }
    
    pub fn unsafe_replace_operand(&mut self, 
        arg:  i32,
        data: *mut c_void)  {
        
        todo!();
        /*
            operands_[arg].data = data;
        */
    }
    
    pub fn narrow(&mut self, 
        dim:   i32,
        start: i64,
        size:  i64)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(dim < ndim() && size >= 1);
      shape_[dim] = size;
      view_offsets_[dim] += start;
      for (auto& op : operands_) {
        op.data = ((char*)op.data) + op.stride_bytes[dim] * start;
      }
      if (size == 1 && !is_reduction_) {
        coalesce_dimensions();
      }
        */
    }
    
    pub fn select_all_keeping_dim(&mut self, 
        start_dim: i32,
        indices:   &[i32])  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(start_dim <= ndim());
      for (int i = start_dim; i < ndim(); ++i) {
        for (auto& op : operands_) {
          op.data = ((char*)op.data) + op.stride_bytes[i] * indices[i - start_dim];
        }
        shape_[i] = 1;
      }
        */
    }

    /**
      | Helper to construct a binary op that
      | promotes integer inputs to float.
      |
      */
    pub fn build_binary_float_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
            build(BINARY_FLOAT_OP_CONFIG()
            .add_owned_output(out)
            .add_owned_input(a)
            .add_owned_input(b));
        */
    }
    
    pub fn build_borrowing_binary_float_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
            build(BINARY_FLOAT_OP_CONFIG()
            .add_output(out)
            .add_input(a)
            .add_input(b));
        */
    }
    
    pub fn build_binary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
            build(BINARY_OP_CONFIG()
          .add_owned_output(out)
          .add_owned_input(a)
          .add_owned_input(b));
        */
    }
    
    pub fn build_borrowing_binary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor)  {
        
        todo!();
        /*
            build(BINARY_OP_CONFIG()
          .add_output(out)
          .add_input(a)
          .add_input(b));
        */
    }
    
    pub fn build_unary_float_op(&mut self, 
        out: &Tensor,
        a:   &Tensor)  {
        
        todo!();
        /*
            build(TensorIteratorConfig()
          .set_check_mem_overlap(true)
          .add_owned_output(out)
          .add_owned_input(a)
          .promote_inputs_to_common_dtype(true)
          .cast_common_dtype_to_outputs(true)
          .enforce_safe_casting_to_output(true)
          .promote_integer_inputs_to_float(true));
        */
    }
    
    pub fn build_unary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor)  {
        
        todo!();
        /*
            build(TensorIteratorConfig()
          .set_check_mem_overlap(true)
          .add_owned_output(out)
          .add_owned_input(a)
          .cast_common_dtype_to_outputs(false)
          .enforce_safe_casting_to_output(false)
          .check_all_same_dtype(true));
        */
    }
}

#[macro_export] macro_rules! nullary_op_config {
    () => {
        /*
        
          TensorIteratorConfig()                                        
            .set_check_mem_overlap(true)                                
            .check_all_same_dtype(false)                                
          /* FIXME: workaround for bug: https://github.com/pytorch/pytorch/issues/20342 */ 
            .resize_outputs(false)
        */
    }
}

impl TensorIterator {
    
    pub fn binary_op(&mut self, 
        out: &mut Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TensorIterator iter;
      iter.build_binary_op(out, a, b);
      return iter;
        */
    }
    
    pub fn borrowing_binary_op(&mut self, 
        out: &Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TensorIterator iter;
      iter.build_borrowing_binary_op(out, a, b);
      return iter;
        */
    }
    
    pub fn binary_float_op(&mut self, 
        out: &mut Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TensorIterator iter;
      iter.build_binary_float_op(out, a, b);
      return iter;
        */
    }
    
    pub fn comparison_op(&mut self, 
        out: &mut Tensor,
        a:   &Tensor,
        b:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            // Note [special-case bool outputs]
      // We explicitly don't call `cast_common_dtype_to_outputs` when the output tensor
      // has `bool` dtype. This is a performance optimization: the functional
      // version of all comparison/logical ops uses a bool output tensor, and we'd like to
      // avoid creating a temporary copy of the output.
      // However, note that all kernels using this TensorIterator will need to special-case when
      // the output tensor has bool dtype, and provide a lambda of type (Scalar, Scalar -> bool).
      if (out.scalar_type() == kBool) {
        return TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_owned_output(out)
        .add_owned_input(a)
        .add_owned_input(b)
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .build();
      } else {
        return TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_owned_output(out)
        .add_owned_input(a)
        .add_owned_input(b)
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();
      }
        */
    }
    
    pub fn unary_op(&mut self, 
        out: &mut Tensor,
        a:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TensorIterator iter;
      iter.build_unary_op(out, a);
      return iter;
        */
    }
    
    pub fn unary_float_op(&mut self, 
        out: &mut Tensor,
        a:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TensorIterator iter;
      iter.build_unary_float_op(out, a);
      return iter;
        */
    }
    
    pub fn nullary_op(&mut self, out: &mut Tensor) -> TensorIterator {
        
        todo!();
        /*
            return NULLARY_OP_CONFIG()
        .add_owned_output(out)
        .build();
        */
    }
    
    pub fn borrowing_nullary_op(&mut self, out: &Tensor) -> TensorIterator {
        
        todo!();
        /*
            return NULLARY_OP_CONFIG()
        .add_output(out)
        .build();
        */
    }
    
    pub fn reduce_op(&mut self, 
        out: &mut Tensor,
        a:   &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(out.defined());
      return TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .add_owned_output(out)
        .add_owned_input(a)
        .resize_outputs(false)
        .is_reduction(true)
        // TODO: not supporting casting to outputs is only really necessary for arg{min,max}
        .promote_inputs_to_common_dtype(true)
        .build();
        */
    }
    
    pub fn reduce_op(&mut self, 
        out1: &mut Tensor,
        out2: &mut Tensor,
        a:    &Tensor) -> TensorIterator {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(out1.defined());
      TORCH_INTERNAL_ASSERT(out2.defined());
      TORCH_CHECK(a.device() == out1.device() && out1.device() == out2.device(),
          "reduce_op(): expected input and both outputs to be on same device, but input is on ", a.device(),
          ", output1 is on ", out1.device(), " and output2 is on", out2.device());
      TORCH_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims, but output1 has ", out1.dim(),
          " and output2 has ", out2.dim());
      TORCH_CHECK(out1.sizes() == out2.sizes(), "reduce_op(): expected both outputs to have same sizes, but output1 has ", out1.sizes(),
          " and output2 has ", out2.sizes());
      TORCH_CHECK(out1.strides() == out2.strides(), "reduce_op(): expected both outputs to have same strides, but output1 has ", out1.strides(),
          " and output2 has ", out2.strides());
      return TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .add_owned_output(out1)
        .add_owned_output(out2)
        .add_owned_input(a)
        .resize_outputs(false)
        .is_reduction(true)
        .check_all_same_dtype(false)
        .build();
        */
    }
}

impl TensorIteratorBase {
    
    pub fn populate_operands(&mut self, config: &mut TensorIteratorConfig)  {
        
        todo!();
        /*
            for (auto& tensor: config.tensors_) {
        // If *any* of the arguments is a meta tensor, the overall
        // computation is a meta computation (don't do any work,
        // just compute output information).  This aligns with
        // our multiple dispatch semantics.
        if (tensor->is_meta()) {
          is_meta_ = true;
        }
        operands_.emplace_back(move(tensor));
      }
      num_outputs_ = config.num_outputs_;
        */
    }
    
    pub fn mark_outputs(&mut self)  {
        
        todo!();
        /*
            // TODO: merge this into populate_operands
      for (int i = 0; i < num_outputs_; i++) {
        operands_[i].is_output = true;
        const auto& output = operands_[i].tensor;
        if (!output->defined()) continue;

        // check if output is also an input
        for (int arg = num_outputs_; arg < ntensors(); arg++) {
          const auto& input = operands_[arg].tensor;
          if (output->is_same(*input)) {
            operands_[i].is_read_write = true;
          }
        }
      }
        */
    }
    
    pub fn mark_resize_outputs(&mut self, config: &TensorIteratorConfig)  {
        
        todo!();
        /*
            // Outputs cannot be broadcasted. Check that the shape of the outputs matches
      // the inferred shape. There's an exception for write-only tensors to support
      // our legacy behavior that functions with `out=` arguments resize their
      // outputs.
      if (config.static_shape_.has_value()) {
        return;
      }
      for (int i = 0; i < num_outputs_; i++) {
        const auto& output = operands_[i].tensor;
        if (output->defined() && !output->sizes().equals(shape_)) {
          if (config.resize_outputs_ && !operands_[i].is_read_write) {
            operands_[i].will_resize = true;
            continue;
          }
          // for reduction, output size does not match shape_, as output is reduced size, and shape_ is size of the input
          TORCH_CHECK(is_reduction_,  "output with shape ", output->sizes(), " doesn't match the broadcast shape ",
                     shape_);
        }
      }
        */
    }
    
    pub fn compute_mem_overlaps(&mut self, config: &TensorIteratorConfig)  {
        
        todo!();
        /*
            if (!config.check_mem_overlap_) {
        return;
      }
      for (int i = 0; i < num_outputs_; i++) {
        const auto& output = operands_[i].tensor;
        if (!output->defined()) continue;
        assert_no_internal_overlap(*output);
        for (int j = num_outputs_; j < ntensors(); j++) {
          const auto& input = operands_[j].tensor;
          if (input->unsafeGetTensorImpl()!=output->unsafeGetTensorImpl()) {
            assert_no_partial_overlap(*output, *input);
          }
        }
      }
        */
    }
    
    pub fn compute_shape(&mut self, config: &TensorIteratorConfig)  {
        
        todo!();
        /*
            if (config.static_shape_.has_value()) {
        shape_ = *config.static_shape_;
        return;
      }

      all_ops_same_shape_ = true;
      bool has_scalars = false;
      bool has_tensors = false;
      for (auto& op : operands_) {
        if (!op.tensor->defined()) continue;

        // For now, don't include output tensors when we're resizing outputs.
        // These shapes don't participate in shape computation.
        // This preserves the legacy behavior where torch.add(..., out=dst) resizes
        // the destination tensor.  If the output tensor is also an input, we'll
        // pick it up later in the operands.
        if (config.resize_outputs_ && op.is_output) continue;
        auto shape = op.tensor->sizes();
        if (shape.size() == 0) {
          has_scalars = true;
        } else {
          has_tensors = true;
        }
        if (has_scalars && has_tensors) {
          all_ops_same_shape_ = false;
        }
        if (shape_.empty()) {
          shape_ = shape;
        } else if (!shape.equals(shape_)) {
          all_ops_same_shape_ = false;
          shape_ = infer_size_dimvector(shape_, shape);
        }
      }
        */
    }
    
    pub fn compute_strides(&mut self, config: &TensorIteratorConfig)  {
        
        todo!();
        /*
            for (auto& op : operands_) {
        if (op.tensor->defined()) {
          IntArrayRef original_shape = config.static_shape_ ? shape_ : op.tensor->sizes();
          auto original_stride = op.tensor->strides();
          auto element_size_in_bytes = op.tensor->element_size();
          auto offset = ndim() - original_shape.size();
          if (offset > 0)
              op.stride_bytes.resize(ndim(), 0);
          else
              op.stride_bytes.resize(ndim());
          for (usize i = 0; i < original_shape.size(); i++) {
            // see NOTE: [Computing output strides]
            if (original_shape[i] == 1 && shape_[offset + i] !=1) {
              op.stride_bytes[offset + i] = 0;
            } else {
              op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
            }
          }
        }
      }
        */
    }
    
    pub fn can_use_32bit_indexing(&self) -> bool {
        
        todo!();
        /*
            i64 max_value = i32::max;
      if (numel() > max_value) {
        return false;
      }
      for (auto& op : operands_) {
        i64 max_offset = 1;
        for (int dim = 0; dim < ndim(); dim++) {
          max_offset += (shape_[dim] - 1) * op.stride_bytes[dim];
        }
        if (max_offset > max_value) {
          return false;
        }
      }
      return true;
        */
    }
    
    pub fn split(&mut self, dim: i32) -> Box<TensorIterator> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(dim >= 0 && dim < ndim() && shape()[dim] >= 2);
      unique_ptr<TensorIterator> copy(new TensorIterator(*this));

      bool overlaps = is_dim_reduced(dim);
      auto copy_size = shape_[dim] / 2;
      auto this_size = shape_[dim] - copy_size;
      copy->narrow(dim, 0, copy_size);
      copy->final_output_ &= !overlaps;
      this->narrow(dim, copy_size, this_size);
      this->accumulate_ |= overlaps;

      return copy;
        */
    }
    
    pub fn get_dim_to_split(&self) -> i32 {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(ndim() >= 1);
      i64 max_extent = -1;
      int dim_to_split = -1;
      for (int dim = ndim() - 1; dim >= 0; dim--) {
        const i64 size = shape_[dim];
        if (size == 0) {
          continue;
        }
        for (auto& op : operands_) {
          // abs is necessary to handle some special cases where we support negative strides
          // see the CUDA backend of flip
          const i64 extent = (size - 1) * abs(op.stride_bytes[dim]);
          if (extent > max_extent) {
            max_extent = extent;
            dim_to_split = dim;
          }
        }
      }
      TORCH_INTERNAL_ASSERT(max_extent >= 0);
      return dim_to_split;
        */
    }
    
    pub fn fast_set_up(&mut self, config: &TensorIteratorConfig) -> bool {
        
        todo!();
        /*
            // This function tries to do a fast setup to avoid needless reordering of dimensions and tracking output strides
      // Return true if it can do fast setup or false otherwise
      // TODO enable fast handling for reductions
      FastSetupType setup_type = compute_fast_setup_type(config);
      if (setup_type == FastSetupType::NONE) {
        return false;
      }

      // allocate memory for output, memory format depends on setup_type
      switch (setup_type) {
        case FastSetupType::CONTIGUOUS:
          {
            for (int i = 0; i < num_outputs_; i++){
              auto& op = operands_[i];
              if (!op.tensor->defined()) {
                TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
              }
              set_output(i, shape_, {}, original_options(op).memory_format(MemoryFormat::Contiguous), names_);
            }
            break;
          }
        case FastSetupType::CHANNELS_LAST:
          {
            for (int i = 0; i < num_outputs_; i++){
              auto& op = operands_[i];
              if (!op.tensor->defined()) {
                TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
              }
              set_output(i, shape_, {}, original_options(op).memory_format(MemoryFormat::ChannelsLast), names_);
            }
            break;
          }
        case FastSetupType::NON_OVERLAPPING_DENSE:
          {
            // find the index of a defined tensor in operands_ start from input tensor
            int i_defined; // NOLINT(cppcoreguidelines-init-variables)
            for (i_defined = ntensors() - 1; i_defined >= 0; --i_defined) {
              if (operands_[i_defined].tensor->defined()) break;
            }
            TORCH_CHECK(i_defined >= 0, "Can not find a defined tensor when fast allocating memory to outputs");
            for (int i = 0; i < num_outputs_; i++){
              auto& op = operands_[i];
              if (!op.tensor->defined()) {
                TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
              }
              set_output(i, shape_, operands_[i_defined].tensor->strides(), original_options(op), names_);
            }
            break;
          }
        default:
          TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", to_string((int)setup_type));
      }
      //coalescing dimensions consists of collapsing dimensions to 1 (we are limited to contiguous no-broadcast cases here)
      if (ndim() > 1){
        has_coalesced_dimensions_ = true;
      }
      if (ndim() >= 1) {
        shape_[0] = numel();
        shape_.resize(1);
      }
      for (auto& op : operands_ ) {
        auto element_size_in_bytes = op.tensor->element_size();
        op.stride_bytes.resize(ndim());
        if (ndim()>0) {
          op.stride_bytes[0] = element_size_in_bytes;
        }
      }
      return true;
        */
    }
    
    pub fn compute_fast_setup_type(&mut self, config: &TensorIteratorConfig) -> FastSetupType {
        
        todo!();
        /*
            if (is_reduction_ || !all_ops_same_shape_) {
        return FastSetupType::NONE;
      }

      // For linear iteration, only contiguous tensors can be coalesced
      // Fast setup of any other format requires changing iteration order
      if (enforce_linear_iteration_) {
        for (const auto& op : operands_) {
          if (op.tensor->defined() && !op.will_resize) {
            auto is_contiguous = op.tensor->is_contiguous(MemoryFormat::Contiguous);
            if (!is_contiguous) {
              return FastSetupType::NONE;
            }
          }
        }
        return FastSetupType::CONTIGUOUS;
      }

      bool is_contiguous = true;
      bool is_channels_last = true;
      bool is_non_overlapping_and_dense = true;
      for (const auto& op : operands_) {
        if (op.tensor->defined() && !op.will_resize) {
          is_contiguous &= op.tensor->is_contiguous(MemoryFormat::Contiguous);
          is_channels_last &= op.tensor->is_contiguous(MemoryFormat::ChannelsLast);
          is_non_overlapping_and_dense &= op.tensor->is_non_overlapping_and_dense();
        }
      }
      // TODO this leads to ambiguous cases (NC11) to be always treated as contiguous
      if (is_contiguous) {
        return FastSetupType::CONTIGUOUS;
      }
      if (is_channels_last) {
        return FastSetupType::CHANNELS_LAST;
      }
      if (is_non_overlapping_and_dense) {
        i64 prev = -1;
        // Fast setup is allowed only when all the defined tensors have the same shape and strides,
        // Iterate from back to check input tensors' strides first, then output tensors'.
        for (i64 i = ntensors() - 1; i >= 0; --i) {
          const auto& op = operands_[i];
          if (op.tensor->defined() && !op.will_resize) {
            if (prev < 0) {
              prev = i;
              continue;
            }
            if (!operands_[prev].tensor->strides().equals(op.tensor->strides())) {
              // [Note: stride check for non contiguous tensors in fast setup]
              // We prevent 3 cases doing fast setup here:
              // 1. input tensors have different strides.
              // 2. output tensors won't be resized and have different strides.
              // 3. input tensors have the same strides, but output tensors have different strides with input tensors.
              //    We don't allow re-stride output tensors in this case since it is not compatible with
              //    numpy. The behavior in numpy is that if the output tensor has same shape as the input
              //    tensor but different strides, the strides of output tensor will be preserved, so we do
              //    the same in tensor iterator.
              return FastSetupType::NONE;
            }
          }
        }
        return FastSetupType::NON_OVERLAPPING_DENSE;
      }
      return FastSetupType::NONE;
        */
    }
    
    pub fn build(&mut self, config: &mut TensorIteratorConfig)  {
        
        todo!();
        /*
            // populate some persistent configuration fields
      is_reduction_ = config.is_reduction_;
      enforce_linear_iteration_ = config.enforce_linear_iteration_;

      // fill in operands_ based on configuration
      populate_operands(config);
      // set is_output and is_read_write flags on appropriate tensors
      mark_outputs();
      // Check that the outputs have no internal overlap
      // and do not share memory with inputs.
      compute_mem_overlaps(config);
      // Check that input dimensions are aligned correctly & compute outnames.
      compute_names(config);
      // compute the broadcasted shape
      compute_shape(config);
      // mark outputs for resizing if necessary
      mark_resize_outputs(config);
      // compute the result dtype and device
      compute_types(config);
      // try fast setup output tensor, if failed, fallback to normal setup
      if (!fast_set_up(config)) {
        // compute each tensor's stride after broadcasting
        compute_strides(config);
        // re-order dimensions to improve coalescing
        reorder_dimensions();
        // allocate the output tensor if it's not provided
        allocate_or_resize_outputs();
        // coalesce adjacent dimensions when possible
        if (!is_meta_) coalesce_dimensions();
      }

      if (is_meta_) return;

      // XLA tensors don't have storage, so they don't have an underlying data pointer.
      // Nothing beyond this point is important for meta functions, so it's fine to exit early here.
      if (common_device_.type() == DeviceType_XLA) return;

      for (auto& op : operands_) {
        TORCH_INTERNAL_ASSERT(op.tensor->defined());
        op.data = op.tensor->data_ptr();
      }

      // zero out offsets
      // If the tensor is a scalar, we leave room for it
      // So index translations in reduction can access
      // a valid value for the offset
      i64 ndim_offsets = (ndim() ? ndim() : 1);
      view_offsets_ = DimVector(ndim_offsets, 0);
        */
    }

    /**
      | This is the structured kernels implementation
      | of set_output.
      |
      | It is NEVER actually called directly; instead,
      | a subclass of TensorIteratorBase will override
      | set_output to actually do the operation, and
      | then call set_output on the TensorIteratorBase
      | to setup TI's metadata.
      |
      | The precondition for this function is that
      | maybe_get_output() now unconditionally returns
      | a real Tensor (prior to output setting, this
      | function may return an undefined tensor.)
      |
      */
    pub fn set_output(&mut self, 
        output_idx: i64,
        sizes:      &[i32],
        strides:    &[i32],
        options:    TensorOptions,
        names:      DimnameList)  {
        
        todo!();
        /*
            auto& op = operands_[output_idx];
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
      const auto& t = maybe_get_output(output_idx);
      TORCH_INTERNAL_ASSERT(t.defined());
      if (!op.tensor->defined()) {
        op.tensor = MaybeOwned<Tensor>::borrowed(t);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(op.target_dtype == t.scalar_type());
      } else if (op.will_resize) {
        if (op.original_tensor->defined()) {
          // OK, so this is pretty weird.  To understand how we can end up in
          // this situation, first look at Marker [Output original_tensor is set].
          // That is the sole site where original_tensor may be set on an
          // output operand.  Essentially, when we are given an explicit output
          // tensor whose dtype doesn't match the computed common dtype from
          // the input operands, we do a switcheroo: we replace the (incorrectly
          // typed) output tensor with a correctly typed, *temporary* tensor,
          // and remember the original tensor in original_tensor (which will
          // then get written back to when we cast_outputs).
          //
          // Now, what if the given output tensor also happened to be zero
          // size (meaning that we will_resize it)?  Well, at the call site
          // above, we don't necessarily(*) know what the correct shape should
          // be, so we give the temporary tensor the same shape as the original.
          // At the time of set_output is when we DO know what the correct size
          // is, and the subclass's implementation of set_output in structured class
          // responsible for resizing original_tensor.  But we still have this
          // incorrectly sized temporary output which the structured subclass
          // knows nothing about, so we are obligated to also resize it here.
          //
          // This is a slight memory pessimization, because previously
          // original_tensor only got resized at the end of the computation, rather
          // than at the beginning (as happens here).  However, the peak memory
          // usage is the same, since you need to materialize both original tensor
          // and temporary tensor to do the copy.
          //
          // (*) Actually, technically, we probably do know what the shape
          // should be, since we do shape computation before dtype computation.
          // So hypothetically we could figure out what the correct shape is
          // at that point in time and directly allocate the temporary at
          // the right size.
          //
          // But a better solution is to delay allocation of temporaries until
          // after TensorIterator builder, waiting until we actually want
          // to do the computation.  That would also remove the necessity
          // for the is_meta_ test.
          TORCH_INTERNAL_ASSERT(op.original_tensor->is_same(t));
          TORCH_INTERNAL_ASSERT(!op.tensor->is_same(t));
          native::resize_output(*op.tensor, sizes);
          if (!strides.empty()) {
            TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
            op.tensor->as_strided_(sizes, strides);
          } else if (options.memory_format_opt().has_value()) {
            op.tensor->unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
          }
        }
      }
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          op.tensor->is_same(t) || op.current_dtype == op.tensor->scalar_type());
    // For simplicity, just always update the cached current_type.
      op.current_dtype = op.tensor->scalar_type();
        */
    }
}

impl TensorIterator {
    
    /**
      | This is the "traditional" implementation of
      | set_output.  On TensorIterator instances, it is
      | invoked directly from various call sites in
      | this file.  No funny business.
      */
    pub fn set_output(&mut self, 
        output_idx: i64,
        sizes:      &[i32],
        strides:    &[i32],
        options:    TensorOptions,
        names:      DimnameList)  {
        
        todo!();
        /*
            // NB: intentionally no superclass call
      auto& op = operands_[output_idx];
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
      if (!op.tensor->defined()) {
          if (strides.empty()) {
            op.tensor = MaybeOwned<Tensor>::owned(empty(sizes, options));
          } else {
            op.tensor = MaybeOwned<Tensor>::owned(empty_strided(sizes, strides, options));
          }
          op.current_dtype = op.target_dtype;
      } else if (op.will_resize) {
          native::resize_output(*op.tensor, sizes);
          if (!strides.empty()) {
            TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
            op.tensor->as_strided_(sizes, strides);
          } else if (options.memory_format_opt().has_value()) {
            op.tensor->unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
          }
      }
      if (!names.empty()) {
        TORCH_INTERNAL_ASSERT(op.tensor->defined());
        namedinference::propagate_names(*op.tensor, names);
      }
        */
    }

    /**
      | Not actually used by anything (TensorIterator
      | subclass calls its own implementation of
      | set_output which knows exactly where all the
      | outputs are), but we have to provide all pure
      | virtual methods for MetaBase
      |
      */
    pub fn maybe_get_output(&mut self, output_idx: i64) -> &Tensor {
        
        todo!();
        /*
            return *operands_[output_idx].tensor;
        */
    }
}

impl TensorIteratorBase {
    
    pub fn with_32bit_indexing(&self) -> SplitUntil32Bit {
        
        todo!();
        /*
            return SplitUntil32Bit(*this);
        */
    }
}

impl SplitUntil32BitIterator {

    /**
      | SplitUntil32Bit. Recursively splits an
      | iterator into sub-iterators that can use
      | 32-bit indexing.
      |
      */
    pub fn new(iter: &TensorIteratorBase) -> Self {
    
        todo!();
        /*


            vec.emplace_back(new TensorIterator(iter));
      vec.emplace_back(nullptr); // ++ first pops the last element
      ++(*this);
        */
    }
    
    pub fn prefix_increment(&mut self) -> &mut SplitUntil32BitIterator {
        
        todo!();
        /*
            vec.pop_back();
      while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
        auto& iter = *vec.back();
        i64 split_dim = iter.get_dim_to_split();
        vec.emplace_back(iter.split(split_dim));
      }
      return *this;
        */
    }
    
    pub fn operator_star(&self) -> &mut TensorIterator {
        
        todo!();
        /*
            return *vec.back();
        */
    }
}

impl SplitUntil32Bit {
    
    pub fn begin(&self) -> SplitUntil32BitIterator {
        
        todo!();
        /*
            return SplitUntil32Bit::iterator(iter);
        */
    }
    
    pub fn end(&self) -> SplitUntil32BitIterator {
        
        todo!();
        /*
            return SplitUntil32Bit::iterator();
        */
    }
}

impl DimCounter {
    
    pub fn new(
        shape: &[i32],
        range: Range) -> Self {
    
        todo!();
        /*


            : shape(shape)
      , range(range)
      , values(shape.size())
      , offset(range.begin) 
      fill(values.begin(), values.end(), 0);
      if (range.begin == 0) {
        return;
      }

      i64 linear_offset = range.begin;
      i64 ndim = values.size();
      for (const auto dim : irange(ndim)) {
        i64 size = shape[dim];
        if (size > 0) {
          values[dim] = linear_offset % size;
          linear_offset /= size;
        }
      }
      TORCH_INTERNAL_ASSERT(linear_offset == 0);
        */
    }
    
    pub fn is_done(&self) -> bool {
        
        todo!();
        /*
            return offset >= range.end;
        */
    }
    
    pub fn increment(&mut self, step: &[i64; 2])  {
        
        todo!();
        /*
            offset += step[0] * step[1];
      i64 ndim = values.size();
      i64 overflow = step[0];
      int i = 0;
      if (step[1] != 1) {
        TORCH_INTERNAL_ASSERT(step[0] == shape[0] && values[0] == 0);
        i = 1;
        overflow = step[1];
      }
      for (; i < ndim && overflow > 0; i++) {
        auto size = shape[i];
        auto prev = values[i];
        auto value = prev + overflow;
        if (value >= size) {
          overflow = 1;
          value -= size;
          TORCH_INTERNAL_ASSERT(value < size);
        } else {
          overflow = 0;
        }
        values[i] = value;
      }
      TORCH_INTERNAL_ASSERT(overflow == 0 || overflow == 1);
        */
    }
    
    pub fn max_2d_step(&self) -> [i64; 2] {
        
        todo!();
        /*
            i64 step0 = min(shape[0] - values[0], range.end - offset);
      i64 step1 = 1;
      if (step0 == shape[0] && shape.size() >= 1) {
        step1 = min(shape[1] - values[1], (range.end - offset) / shape[0]);
      }
      return {step0, step1};
        */
    }
}
