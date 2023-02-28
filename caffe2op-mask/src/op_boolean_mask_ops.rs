crate::ix!();

use crate::{
    GradientMakerBase,
    Workspace,
    OperatorDef,
    OperatorStorage,
    CPUContext
};

#[test] fn boolean_mask_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BooleanMask",
        ["data", "mask"],
        ["masked_data", "masked_indices"]
    )

    workspace.FeedBlob("data", np.array([1,2,3,4,5,6]))
    workspace.FeedBlob("mask", np.array([True,False,False,True,True,False]))
    print("data:", workspace.FetchBlob("data"))
    print("mask:", workspace.FetchBlob("mask"))
    workspace.RunOperatorOnce(op)
    print("masked_data:", workspace.FetchBlob("masked_data"))
    print("masked_indices:", workspace.FetchBlob("masked_indices"))



    result:
    data: [1 2 3 4 5 6]
    mask: [ True False False  True  True False]
    masked_data: [1 4 5]
    masked_indices: [0 3 4]
    */
}

/**
  | Given a 1D `data` tensor and a boolean
  | `mask` tensor of the same shape, returns
  | a `masked_data` tensor containing
  | only the elements corresponding to
  | positions where the `mask` is True,
  | and a `masked_indices` tensor containing
  | the indices of the True elements.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc
  |
  */
pub struct BooleanMaskOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{BooleanMask, 2}

num_outputs!{BooleanMask, (1,2)}

inputs!{
    BooleanMask, 
    0 => ("data", "(*Tensor*): 1D input tensor"),
    1 => ("mask", "(*Tensor`<bool>`*): tensor of bools which determines the input elements that will be left in the `masked_data` output tensor; same shape as `data`")
}

outputs!{
    BooleanMask, 
    0 => ("masked_data", "(*Tensor*): 1D tensor of same type as `data` input that contains the masked input tensor"),
    1 => ("masked_indices", "(*Tensor`<int>`*): 1D tensor of indices of the True elements in the `mask` tensor")
}

impl<Context> BooleanMaskOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

///-------------------------------------------
pub struct BooleanMaskOpGradient<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

impl<Context> BooleanMaskOpGradient<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    /**
      | Calculating the gradient of the Boolean
      | Mask operator requires access to the
      | original mask that's passed in, and
      | the gradient to backpropagate.
      |
      */
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, std::int32_t, std::int64_t, float, double>>::
            call(this, Input(1));
        */
    }
}

impl BooleanMaskOpGradient<CPUContext> {

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& mask = Input(0);
          const auto& dY = Input(1);
          auto* dX = Output(0);

          const int data_length_before_mask = mask.size(0);

          dX->Resize(data_length_before_mask);

          // TODO: we should support any type, not just float
          T* dXdata = dX->template mutable_data<T>();
          const T* dYdata = dY.template data<T>();
          const bool* mask_data = mask.template data<bool>();

          int ind = 0;

          for (int i = 0; i < data_length_before_mask; i++) {
            dXdata[i] = mask_data[i] ? dYdata[ind++] : 0;
          }

          return true;
        */
    }
}

/**
 | Mask op designed for use in attention mechanisms
 | for sequence modeling tasks.
 |
 | Supports batching: given batch_dim, collapses dims
 | 0 through batch_dim into a single dimension,
 | e.g. if tensor dims are [4,2,1,3,4] and
 | batch_dim=2, first collapse tensor to [4*2*1,3,4],
 | then mask each batch [i,:,:].
 |
 | Two current operating modes:
 |
 | 1) Given a 2D input tensor and 1D tensor of
 | sequence lengths, for each row i in the input
 | tensor, set elements in that row to -inf if their
 | column index j >= sequence_lengths[i]. This mode
 | takes two inputs and argument mode = 'sequence'
 |
 | 2) Triangular mask. Given row index i and column
 | index j, set elements to -inf given the following
 | conditions:
 |
 |       mode='upper', x_ij = -inf if j < i
 |       mode='lower', x_ij = -inf if j > i
 |       mode='upperdiag', x_ij = -inf if j <= i
 |       mode='lowerdiag', x_ij = -inf if j >= i
 |
 | This mode takes one input.
 |
 | 3) Window Mask. Given a 2D input tensor and 1D
 | tensor of window centers, for each row i in the
 | input tensor, set elements in that row to -inf if
 | their column index j outside [center - radius,
 | center + radius].
 |
 | This mode takes two inputs and argument mode
 | = 'sequence'.
 |
 | Argument 'radius' should be provided.
 */
pub struct SequenceMaskOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    axis:         i32,
    radius:       i32,
    mode:         String,
    grad:         bool,
    fill_val:     f32,
    batch:        i32,
    repeat_from:  i32,
}

register_cpu_operator!{
    SequenceMask, 
    SequenceMaskOp<CPUContext>
}

num_inputs!{SequenceMask, (1,2)}

num_outputs!{SequenceMask, 1}

inputs!{SequenceMask, 
    0 => ("input",             "Tensor to apply masking to"),
    1 => ("sequence_lengths",  "1D Tensor of sequence lengths for mode #1")
}

outputs!{SequenceMask, 
    0 => ("masked_tensor", "Input tensor with masking applied")
}

args!{SequenceMask, 
    0 => ("mode",             "(string) Mode selection. Possible values: 'sequence', 'upper', 'lower', 'upperdiag', 'lowerdiag'"),
    1 => ("axis",             "(int) Beginning axis of row elements. All dimensions to the left will be treated as row indices and those to the right (inclusive) will be treated as column indices in the 2D mask"),
    2 => ("grad",             "(bool) operate in gradient mode"),
    3 => ("radius",           "(int) radius of windows in window mode"),
    4 => ("batch",            "(int) batch dimension of tensor (optional)"),
    5 => ("repeat_from_axis", "(int) used when mask should be repeated for one or more data dimensions (beginning at this axis).  (currently only supported for sequence mode without batch argument)")
}

impl<Context> SequenceMaskOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            axis_(this->template GetSingleArgument<int>("axis", 1)),
            radius_(this->template GetSingleArgument<int>("radius", 10)),
            grad_(this->template GetSingleArgument<bool>("grad", false)),
            fill_val_(this->template GetSingleArgument<float>(
                    "fill_val",
                    -1.0f * std::numeric_limits<float>::infinity())) 

                // Mode argument is required
                mode_ = GetArgument(operator_def, "mode").s();
            // batch argument is optional, but if not given, we don't want a default val
            if (HasArgument("batch")) {
                batch_ = GetArgument(operator_def, "batch").i();
            }

            if (HasArgument("repeat_from_axis")) {
                CAFFE_ENFORCE(
                    mode_ == "sequence",
                    "repeat_from_axis currently only supported in sequence mode.");
                CAFFE_ENFORCE(
                    !HasArgument("batch"),
                    "repeat_from_axis and batch not currently supported together.");
                repeat_from_ =
                    this->template GetSingleArgument<int>("repeat_from_axis", -1);
            }
        */
    }
}

#[test] fn boolean_mask_lengths_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BooleanMaskLengths",
        ["lengths", "mask"],
        ["masked_lengths"]
    )

    workspace.FeedBlob("lengths", np.array([1,3,2], dtype=np.int32))
    workspace.FeedBlob("mask", np.array([False,True,True,False,True,True]))
    print("lengths:", workspace.FetchBlob("lengths"))
    print("mask:", workspace.FetchBlob("mask"))
    workspace.RunOperatorOnce(op)
    print("masked_lengths:", workspace.FetchBlob("masked_lengths"))



    lengths: [1 3 2]
    mask: [False  True  True False  True  True]
    masked_lengths: [0 2 2]
    */
}

/**
  | Given a tensor of int32 `lengths` tensor
  | representing segment lengths and a
  | `mask` (boolean) tensor, return the
  | segment lengths of the corresponding
  | segmented tensor after *BooleanMask**
  | is applied.
  | 
  | If `lengths` tensor is $[a_1, a_2, ...,
  | a_n]$, then length of `mask` tensor
  | must be $a_1 + a_2 + ... + a_n$.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc
  |
  */
pub struct BooleanMaskLengthsOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{BooleanMaskLengths, 2}

num_outputs!{BooleanMaskLengths, 1}

inputs!{
    BooleanMaskLengths, 
    0 => ("lengths", "(*Tensor`<int>`*): input tensor containing segment lengths"),
    1 => ("mask", "(*Tensor`<bool>`*): A 1D bool tensor of values to keep.")
}

outputs!{
    BooleanMaskLengths, 
    0 => ("masked_lengths", "(*Tensor`<int>`*): 1D tensor of same type as inputs that contains the sequence")
}

impl<Context> BooleanMaskLengthsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& lengths = Input(0);
            auto& mask = Input(1);

            CAFFE_ENFORCE(lengths.dim() == 1);
            CAFFE_ENFORCE(mask.dim() == 1);
            const auto* lengthsPtr = lengths.template data<T>();
            const auto* maskPtr = mask.template data<bool>();
            auto totalLength =
                std::accumulate(lengthsPtr, lengthsPtr + lengths.numel(), 0);
            CAFFE_ENFORCE(mask.numel() == totalLength);
            auto* lengthsOut = Output(0, lengths.sizes(), at::dtype<T>());
            auto* lengthsOutPtr = lengthsOut->template mutable_data<T>();
            int p = 0;
            for (int i = 0; i < lengths.numel(); ++i) {
              T lengthOut = 0;
              for (int j = 0; j < lengthsPtr[i]; ++j) {
                if (maskPtr[p++]) {
                  ++lengthOut;
                }
              }
              lengthsOutPtr[i] = lengthOut;
            }
            return true;
        */
    }
}


impl BooleanMaskLengthsOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(0);
      auto& mask = Input(1);
      auto* dataOut = Output(0);
      CAFFE_ENFORCE(data.dim() >= 1);
      CAFFE_ENFORCE_EQ(mask.dim(), 1);
      CAFFE_ENFORCE(data.size(0) == mask.size(0));

      const auto* maskPtr = mask.template data<bool>();
      int numOutputs = 0;
      int outerSize = mask.numel();
      for (int i = 0; i < outerSize; ++i) {
        if (maskPtr[i]) {
          ++numOutputs;
        }
      }
      std::vector<int64_t> outShape;
      outShape.push_back(numOutputs);
      outShape.insert(outShape.end(), data.sizes().begin() + 1, data.sizes().end());
      dataOut->Resize(outShape);
      auto* outPtr = (char*)dataOut->raw_mutable_data(data.dtype());

      int64_t* out_vec = nullptr;
      if (OutputSize() == 2) {
        auto* indicesOut = Output(1, {numOutputs}, at::dtype<int64_t>());
        out_vec = indicesOut->template mutable_data<int64_t>();
      }

      if (numOutputs == 0) {
        return true;
      }
      const auto innerSize = data.size_from_dim(1);
      const auto innerSizeBytes = innerSize * data.dtype().itemsize();

      int64_t lastStart = -1;
      const auto* inPtr = (char*)data.raw_data();
      int64_t outStart = 0;

      for (int64_t i = 0;; ++i) {
        // mask was true and either a) became false, or b) sequence finished
        if (lastStart != -1 && ((i >= outerSize) || !maskPtr[i])) {
          const auto* src = inPtr + lastStart * innerSizeBytes;
          auto* dst = outPtr + outStart * innerSizeBytes;
          int numItems = i - lastStart;
          context_.CopyItemsSameDevice(
              data.dtype(), numItems * innerSize, src, dst);
          outStart += numItems;
          lastStart = -1;
        }
        if (i >= outerSize) {
          break;
        }
        // mask was false and became true
        if (lastStart == -1 && maskPtr[i]) {
          lastStart = i;
        }
        if (maskPtr[i] && OutputSize() == 2) {
          *(out_vec++) = i;
        }
      }
      return true;
        */
    }
}

register_cpu_operator!{
    BooleanMask, 
    BooleanMaskOp<CPUContext>
}

register_cpu_gradient_operator!{ 
    BooleanMaskGradient, 
    BooleanMaskOpGradient<CPUContext>
}

register_cpu_operator!{
    BooleanMaskLengths, 
    BooleanMaskLengthsOp<CPUContext>
}

num_inputs!{BooleanMaskGradient, 2}

num_outputs!{BooleanMaskGradient, 1}

pub struct GetBooleanMaskGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetBooleanMaskGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BooleanMaskGradient",
            "",
            vector<string>{I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{BooleanMask, GetBooleanMaskGradient}

no_gradient!{BooleanMaskLengths}

pub const minf: f32 = -1.0 * f32::INFINITY;

/**
  | Template this on a functor object so
  | we can generate different implementations
  | at compile time and have a better chance
  | of inlining
  |
  */
#[inline] pub fn mask_with_functor<Functor>(
    n:        i32,
    m:        i32,
    b:        i32,
    input:    *const f32,
    func:     Functor,
    fill_val: f32,
    out:      *mut f32) 
{
    todo!();
    /*
       if (B >= 0) { // with batching
        // collapse tensor to 3-dim view [B, N, M] where:
        // B is product of dims up to and including batch
        // N is product of dims between batch and axis, exclusive
        // M is product of dimensions at/after axis
        // then mask each batch [i, :, :] (note that this is N x M matrix)
        for (int i = 0; i < B; ++i) {
          for (int j = 0; j < N; ++j) {
            for (int k = 0; k < M; ++k) {
              // when [i, :, :] is laid out in row major order
              // N * M * i + M * j + k is index of entry in N x M matrix
              // with coordinates (row = j, col = k)
              auto val = in[N * M * i + M * j + k];
              out[N * M * i + M * j + k] = (fn(j, k, val) ? fill_val : val);
            }
          }
        }
      } else { // without batching
        // TODO(T20952436): vector implementation
        // collapse tensor to 2-dim view [N, M], where
        // N is product of dimensions before axis
        // M is product of dimensions at/after axis
        // and mask N by M matrix
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            auto val = in[M * i + j];
            out[M * i + j] = (fn(i, j, val) ? fill_val : val);
          }
        }
      }
    */
}

/**
  | Repeat masking along continuous segments
  | (right axes) of size D
  |
  */
#[inline] pub fn repeated_mask_with_functor<Functor>(
    n:        i32,
    m:        i32,
    d:        i32,
    input:    *const f32,
    func:     Functor,
    fill_val: f32,
    out:      *mut f32)
{
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          for (int k = 0; k < D; ++k) {
            auto val = in[M * D * i + D * j + k];
            out[M * D * i + D * j + k] = (fn(i, j, val) ? fill_val : val);
          }
        }
      }
    */
}

pub struct SequenceFunctor {
    sl: *const i32,
    len: usize,
}

impl SequenceFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(i < len_, "Out of bound.");
        return j >= sl_[i];
        */
    }
}

pub struct WindowFunctor {
    c: *const i32,
    r: i32,
}

impl WindowFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j > c[i] + r || j < c[i] - r;
        */
    }
}

pub struct UpperFunctor {

}

impl UpperFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j > i;
        */
    }
}

pub struct LowerFunctor {

}

impl LowerFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j < i;
        */
    }
}

pub struct UpperDiagFunctor {

}

impl UpperDiagFunctor {
    
    #[inline] pub fn invoke(
        &mut self, 
        i: i32, 
        j: i32, 
        val: f32) -> bool 
    {
        todo!();
        /*
            return j >= i;
        */
    }
}

pub struct LowerDiagFunctor;

impl LowerDiagFunctor {

    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j <= i;
        */
    }
}

impl SequenceMaskOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<>(&mut self) -> bool {
        todo!();
        /*
            const Tensor* input = &Input(0);
          const Tensor* sequence_lengths = nullptr;
          const Tensor* window_centers = nullptr;

          if (mode_ == "sequence") {
            sequence_lengths = &Input(1);
          } else if (mode_ == "window") {
            window_centers = &Input(1);
          }

          auto* output = Output(0, input->sizes(), at::dtype<T>());

          const auto canonical_axis = input->canonical_axis_index(axis_);

          // canonical_batch is non-negative if batching, -1 otherwise
          int canonical_batch = -1;
          if ((HasArgument("batch"))) {
            canonical_batch = input->canonical_axis_index(batch_);
          }

          // make sure batch < axis
          if (canonical_batch >= 0) {
            CAFFE_ENFORCE_LT(canonical_batch, canonical_axis);
          }

          // if no batch, then left is product of dims up to axis
          // otherwise, left is product of dims between batch and axis
          const int left =
              (canonical_batch >= 0
                   ? input->size_between_dim(canonical_batch, canonical_axis)
                   : input->size_to_dim(canonical_axis));
          const int right = input->size_from_dim(canonical_axis);

          // product of dims from 1 to batch
          const int batch_dim =
              (canonical_batch >= 0
                   ? input->size_to_dim(canonical_batch) * input->size(canonical_batch)
                   : -1);

          T fill_val = convert::To<float, T>(grad_ ? 0.0f : fill_val_);
          if (mode_ == "sequence") {
            CAFFE_ENFORCE(
                sequence_lengths, "Sequence length not provided for mode 'sequence'!");
            if (HasArgument("repeat_from_axis")) {
              const int canonical_repeat_from =
                  input->canonical_axis_index(repeat_from_);
              const int repeated_dims = input->size_from_dim(canonical_repeat_from);
              const int masked_dims = right / repeated_dims;
              RepeatedMaskWithFunctor(
                  left,
                  masked_dims,
                  repeated_dims,
                  input->data<T>(),
                  SequenceFunctor(
                      sequence_lengths->data<int>(), sequence_lengths->numel()),
                  fill_val,
                  output->template mutable_data<T>());
            } else {
              MaskWithFunctor(
                  left,
                  right,
                  batch_dim,
                  input->data<T>(),
                  SequenceFunctor(
                      sequence_lengths->data<int>(), sequence_lengths->numel()),
                  fill_val,
                  output->template mutable_data<T>());
            }
          } else if (mode_ == "window") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                WindowFunctor(window_centers->data<int>(), radius_),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "upper") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                UpperFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "lower") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                LowerFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "upperdiag") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                UpperDiagFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "lowerdiag") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                LowerDiagFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else {
            CAFFE_ENFORCE(false, "Unsupported mode for SequenceMaskOp!");
            return false;
          }

          return true;
        */
    }
}

pub struct GetSequenceMaskGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSequenceMaskGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<Argument> args;
        args.reserve(Def().arg().size());
        for (const auto& x : Def().arg()) {
          args.push_back(x);
        }
        args.push_back(MakeArgument<bool>("grad", true));
        if (def_.input_size() == 1) {
          return SingleGradientDef(
              "SequenceMask",
              "",
              vector<string>{GO(0)},
              vector<string>{GI(0)},
              args);
        } else {
          return SingleGradientDef(
              "SequenceMask",
              "",
              vector<string>{GO(0), I(1)},
              vector<string>{GI(0)},
              args);
        }
        */
    }
}

impl<'a> CopyArguments for GetSequenceMaskGradient<'a> {

    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{SequenceMask, GetSequenceMaskGradient}
