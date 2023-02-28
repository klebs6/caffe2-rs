crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/RNN.h]

pub type LstmFn = fn(
        _0:  &mut Tensor,
        _1:  &mut Tensor,
        _2:  &mut Tensor,
        _3:  &Tensor,
        _4:  TensorList,
        _5:  TensorList,
        _6:  bool,
        _7:  i64,
        _8:  f64,
        _9:  bool,
        _10: bool,
        _11: bool
) -> ();

pub type RnnFn = fn(
        _0:  &mut Tensor,
        _1:  &mut Tensor,
        _2:  &Tensor,
        _3:  &Tensor,
        _4:  TensorList,
        _5:  bool,
        _6:  i64,
        _7:  f64,
        _8:  bool,
        _9:  bool,
        _10: bool
) -> ();

pub type LstmPackedFn = fn(
        _0:  &mut Tensor,
        _1:  &mut Tensor,
        _2:  &mut Tensor,
        _3:  &Tensor,
        _4:  &Tensor,
        _5:  TensorList,
        _6:  TensorList,
        _7:  bool,
        _8:  i64,
        _9:  f64,
        _10: bool,
        _11: bool
) -> ();

pub type RnnPackedFn = fn(
        _0:  &mut Tensor,
        _1:  &mut Tensor,
        _2:  &Tensor,
        _3:  &Tensor,
        _4:  &Tensor,
        _5:  TensorList,
        _6:  bool,
        _7:  i64,
        _8:  f64,
        _9:  bool,
        _10: bool
) -> ();


declare_dispatch!{lstm_fn, lstm_cudnn_stub}
declare_dispatch!{lstm_fn, lstm_miopen_stub}
declare_dispatch!{rnn_fn, gru_cudnn_stub}
declare_dispatch!{rnn_fn, gru_miopen_stub}
declare_dispatch!{rnn_fn, rnn_tanh_cudnn_stub}
declare_dispatch!{rnn_fn, rnn_tanh_miopen_stub}
declare_dispatch!{rnn_fn, rnn_relu_cudnn_stub}
declare_dispatch!{rnn_fn, rnn_relu_miopen_stub}
declare_dispatch!{lstm_packed_fn, lstm_packed_cudnn_stub}
declare_dispatch!{lstm_packed_fn, lstm_packed_miopen_stub}
declare_dispatch!{rnn_packed_fn, gru_packed_cudnn_stub}
declare_dispatch!{rnn_packed_fn, gru_packed_miopen_stub}
declare_dispatch!{rnn_packed_fn, rnn_tanh_packed_cudnn_stub}
declare_dispatch!{rnn_packed_fn, rnn_tanh_packed_miopen_stub}
declare_dispatch!{rnn_packed_fn, rnn_relu_packed_cudnn_stub}
declare_dispatch!{rnn_packed_fn, rnn_relu_packed_miopen_stub}

#[inline] pub fn check_attributes(
        input:       &Tensor,
        params:      &TensorList,
        hiddens:     &TensorList,
        check_dtype: bool)  {
    let check_dtype: bool = check_dtype.unwrap_or(false);

    todo!();
        /*
            auto input_device = input.device();
      auto input_dtype = input.scalar_type();

      auto check_tensors = [&](const string& name, const Tensor& t) {
        if (!t.defined()) return;
        auto t_device = t.device();
        TORCH_CHECK(input_device == t_device,
                 "Input and ", name, " tensors are not at the same device, found input tensor at ",
                 input_device, " and ", name, " tensor at ", t_device);
        if (check_dtype) {
          auto t_dtype = t.scalar_type();
          TORCH_CHECK(input_dtype == t_dtype,
                   "Input and ", name, " tensors are not the same dtype, found input tensor with ",
                   input_dtype, " and ", name, " tensor with ", t_dtype);
        }
      };

      for (auto h : hiddens) check_tensors("hidden", h);
      for (auto p : params) check_tensors("parameter", p);
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/RNN.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

/**
  | Check if pytorch is compiled with MIOpen.
  |
  */
pub fn use_miopen(
        input:         &Tensor,
        dropout_state: f64) -> bool {
    
    todo!();
        /*
            bool is_miopen_acceptable = ((input.scalar_type() == kFloat)|| (input.scalar_type() == kHalf)) &&
                                    (getCUDAHooks().compiledWithMIOpen()) &&
                                    (input.is_cuda()) &&
                                    (dropout_state == 0.0) &&
                                    (globalContext().userEnabledCuDNN());
        return is_miopen_acceptable;
        */
}

/**
  | Those could have been function pointers, but
  | MSVC chokes on function pointers as template
  | parameters
  |
  */
pub struct TanhF {

}

impl TanhF {
    
    pub fn invoke(&self, t: &Tensor) -> Tensor {
        
        todo!();
        /*
            return tanh(t);
        */
    }
}

pub struct ReluF {

}

impl ReluF {
    
    pub fn invoke(&self, t: &Tensor) -> Tensor {
        
        todo!();
        /*
            return relu(t);
        */
    }
}


#[derive(Default)]
pub struct PackedSequence {
    data:        Tensor,
    batch_sizes: Tensor,
}

impl PackedSequence {
    
    pub fn new(
        data:        Tensor,
        batch_sizes: Tensor) -> Self {
    
        todo!();
        /*
        : data(move(_data)),
        : batch_sizes(move(_batch_sizes)),

        
        */
    }
}


/**
  | Simple type for __getstate__/__setstate__
  | serialization
  |
  | Element 0 is a string key to say what kind of
  | CellParam this is. It should be a valid key
  | into cell_params_deserializers
  |
  | Element 1 is the Tensors contained within the
  | CellParams instance
  |
  | Element 2 is the doubles (if any) contained in
  | the CellParams instance
  |
  | Element 3 is the longs (if any) contained
  | within the CellParams instance
  |
  */
pub type CellParamsSerializationType = (
    String,
    Vec<Tensor>,
    Vec<f64>,
    Vec<i64>,
    Vec<IntrusivePtr<LinearPackedParamsBase>>
);

/**
  | Base class so we can polymorphically
  | handle these
  |
  */
pub trait CellParamsBaseInterface:
TorchCustomClassHolder
+ MatmulIh
+ MatmulHh
+ LinearIh
+ LinearHh
+ BIh
+ BHh
+ Getstate {

    /**
      | by default doing nothing. CellParams will
      | override this to define correct behavior for
      | LSTMs with projections.
      |
      | This function is not pure virtual, because
      | it's useful to provide this default
      | implementation, so that all cell params that
      | don't support projections work correctly
      | (e.g. QuantizedCellParams variations)
      |
      */
    fn matmul_hr(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return h;
        */
    }
}

pub trait MatmulIh {
    
    fn matmul_ih(&self, input: &Tensor) -> Tensor;
}

pub trait MatmulHh {
    
    fn matmul_hh(&self, h: &Tensor) -> Tensor;
}

pub trait LinearIh {

    fn linear_ih(&self, input_ih: &Tensor) -> Tensor;
}

pub trait LinearHh {
    
    fn linear_hh(&self, input_hh: &Tensor) -> Tensor;
}

pub trait BIh {
    
    fn b_ih(&self) -> &Tensor;
}

pub trait BHh {
    
    fn b_hh(&self) -> &Tensor;
}

pub trait Getstate {
    
    fn getstate(&self) -> CellParamsSerializationType;
}

/**
  | Pretty much all cells we support take the same
  | set of arguments, but threading those
  | 4 arguments manually is really annoying.
  |
  | Their lifetime is externally managed, so we
  | only pass this struct of references
  | around. LSTMs with projections have 5th
  | argument w_hr, for all other models it's always
  | going to be undefined.
  |
  */
pub struct CellParams {
    base: CellParamsBase,
    w_ih: &Tensor,
    w_hh: &Tensor,

    /**
      | optional
      |
      */
    b_ih: &Tensor,


    /**
      | optional
      |
      */
    b_hh: &Tensor,


    /**
      | only defined for LSTMs with projections
      |
      */
    w_hr: &Tensor,
}

impl CellParams {
    
    pub fn new(
        w_ih: &Tensor,
        w_hh: &Tensor,
        b_ih: &Tensor,
        b_hh: &Tensor,
        w_hr: &Tensor) -> Self {
    
        todo!();
        /*
        : w_ih(_w_ih),
        : w_hh(_w_hh),
        : b_ih(_b_ih),
        : b_hh(_b_hh),
        : w_hr(_w_hr),

            }{
        */
    }
    
    pub fn matmul_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return matmul(input, w_ih.t());
        */
    }
    
    pub fn matmul_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return matmul(h, w_hh.t());
        */
    }
    
    pub fn matmul_hr(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            if (w_hr.defined()) {
          return matmul(h, w_hr.t());
        }
        return h;
        */
    }
    
    pub fn linear_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return linear(input, w_ih, b_ih_);
        */
    }
    
    pub fn linear_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return linear(h, w_hh, b_hh_);
        */
    }
    
    pub fn b_ih(&self) -> &Tensor {
        
        todo!();
        /*
            return b_ih_;
        */
    }
    
    pub fn b_hh(&self) -> &Tensor {
        
        todo!();
        /*
            return b_hh_;
        */
    }
    
    pub fn getstate(&self) -> CellParamsSerializationType {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "Not yet implemented");
        */
    }
    
    pub fn setstate(state: CellParamsSerializationType) -> IntrusivePtr<CellParamsBase> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "Not yet implemented");
        */
    }
}

pub struct QuantizedCellParams {
    base: CellParamsBase,
    w_ih:           Tensor,
    w_hh:           Tensor,
    b_ih:           Tensor,
    b_hh:           Tensor,
    packed_ih:      Tensor,
    packed_hh:      Tensor,
    col_offsets_ih: Tensor,
    col_offsets_hh: Tensor,
    scale_ih:       Scalar,
    scale_hh:       Scalar,
    zero_point_ih:  Scalar,
    zero_point_hh:  Scalar,
}

impl QuantizedCellParams {
    
    pub fn new(
        w_ih:           Tensor,
        w_hh:           Tensor,
        b_ih:           Tensor,
        b_hh:           Tensor,
        packed_ih:      Tensor,
        packed_hh:      Tensor,
        col_offsets_ih: Tensor,
        col_offsets_hh: Tensor,
        scale_ih:       &Scalar,
        scale_hh:       &Scalar,
        zero_point_ih:  &Scalar,
        zero_point_hh:  &Scalar) -> Self {
    
        todo!();
        /*


            : w_ih(move(_w_ih)),
            w_hh(move(_w_hh)),
            b_ih_(move(_b_ih)),
            b_hh_(move(_b_hh)),
            packed_ih(move(_packed_ih)),
            packed_hh(move(_packed_hh)),
            col_offsets_ih(move(_col_offsets_ih)),
            col_offsets_hh(move(_col_offsets_hh)),
            scale_ih(move(_scale_ih)),
            scale_hh(move(_scale_hh)),
            zero_point_ih(move(_zero_point_ih)),
            zero_point_hh(move(_zero_point_hh))
        */
    }
    
    pub fn matmul_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(false, "matmul is not supported with quantized cell params");
        */
    }
    
    pub fn matmul_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(false, "matmul is not supported with quantized cell params");
        */
    }
    
    pub fn linear_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return fbgemm_linear_int8_weight_fp32_activation(
            input, w_ih, packed_ih, col_offsets_ih, scale_ih, zero_point_ih, b_ih_);
        */
    }
    
    pub fn linear_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return fbgemm_linear_int8_weight_fp32_activation(
            h, w_hh, packed_hh, col_offsets_hh, scale_hh, zero_point_hh, b_hh_);
        */
    }
    
    pub fn b_ih(&self) -> &Tensor {
        
        todo!();
        /*
            return b_ih_;
        */
    }
    
    pub fn b_hh(&self) -> &Tensor {
        
        todo!();
        /*
            return b_hh_;
        */
    }
    
    pub fn getstate(&self) -> CellParamsSerializationType {
        
        todo!();
        /*
            vector<Tensor> tensors_to_serialize = {
            w_ih, w_hh, b_ih_, b_hh_, col_offsets_ih, col_offsets_hh};
        vector<double> doubles_to_serialize = {scale_ih.toDouble(),
                                                    scale_hh.toDouble()};
        vector<i64> longs_to_serialize = {zero_point_ih.toLong(),
                                                   zero_point_hh.toLong()};
        return CellParamsSerializationType(
            "quantized",
            move(tensors_to_serialize),
            move(doubles_to_serialize),
            move(longs_to_serialize),
            {});
        */
    }
    
    pub fn setstate(state: CellParamsSerializationType) -> IntrusivePtr<CellParamsBase> {
        
        todo!();
        /*
            vector<Tensor> tensors;
        vector<double> doubles;
        vector<i64> longs;
        tie(ignore, tensors, doubles, longs, ignore) =
            move(state);
        TORCH_INTERNAL_ASSERT(tensors.size() == 6);
        TORCH_INTERNAL_ASSERT(doubles.size() == 2);
        TORCH_INTERNAL_ASSERT(longs.size() == 2);

        Tensor qw_ih = move(tensors[0]), qw_hh = move(tensors[1]),
                   b_ih = move(tensors[2]), b_hh = move(tensors[3]),
                   col_offsets_ih = move(tensors[4]),
                   col_offsets_hh = move(tensors[5]);
        double scale_ih = doubles[0], scale_hh = doubles[1];
        i64 zero_point_ih = longs[0], zero_point_hh = longs[1];

        Tensor packed_ih = native::fbgemm_pack_quantized_matrix(qw_ih);
        Tensor packed_hh = native::fbgemm_pack_quantized_matrix(qw_hh);

        return make_intrusive<QuantizedCellParams>(
            /*w_ih=*/move(qw_ih),
            /*w_hh=*/move(qw_hh),
            /*b_ih_=*/move(b_ih),
            /*b_hh_=*/move(b_hh),
            /*packed_ih=*/move(packed_ih),
            /*packed_hh=*/move(packed_hh),
            /*col_offsets_ih=*/move(col_offsets_ih),
            /*col_offsets_hh=*/move(col_offsets_hh),
            /*scale_ih=*/move(scale_ih),
            /*scale_hh=*/move(scale_hh),
            /*zero_point_ih=*/move(zero_point_ih),
            /*zero_point_hh=*/move(zero_point_hh));
        */
    }
}

pub fn make_quantized_cell_params(
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: Tensor,
    b_hh: Tensor) -> IntrusivePtr<CellParamsBase> {
    
    todo!();
        /*
            auto make_vals = [&](const Tensor& W) {
        auto params = native::fbgemm_linear_quantize_weight(W);
        Tensor packed_weight =
            native::fbgemm_pack_quantized_matrix(get<0>(params));
        return tuple_cat(
            make_tuple(move(packed_weight)), move(params));
      };

      Tensor qw_ih, qw_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh;
      Scalar scale_ih, scale_hh, zero_point_ih, zero_point_hh;

      tie(packed_ih, qw_ih, col_offsets_ih, scale_ih, zero_point_ih) =
          make_vals(w_ih);
      tie(packed_hh, qw_hh, col_offsets_hh, scale_hh, zero_point_hh) =
          make_vals(w_hh);

      return make_intrusive<QuantizedCellParams>(
          /*qw_ih=*/move(qw_ih),
          /*qw_hh=*/move(qw_hh),
          /*b_ih=*/move(b_ih),
          /*b_hh=*/move(b_hh),
          /*packed_ih=*/move(packed_ih),
          /*packed_hh=*/move(packed_hh),
          /*col_offsets_ih=*/move(col_offsets_ih),
          /*col_offsets_hh=*/move(col_offsets_hh),
          /*scale_ih=*/move(scale_ih),
          /*scale_hh=*/move(scale_hh),
          /*zero_point_ih=*/move(zero_point_ih),
          /*zero_point_hh=*/move(zero_point_hh));
        */
}

pub struct QuantizedCellParamsDynamic {
    base:         CellParamsBase,
    packed_w_ih:  IntrusivePtr<LinearPackedParamsBase>,
    packed_w_hh:  IntrusivePtr<LinearPackedParamsBase>,
    b_ih:         Tensor,
    b_hh:         Tensor,
    reduce_range: bool,
}

impl QuantizedCellParamsDynamic {

    pub fn new(
        /* Prepacked Weight Tensor */
        packed_w_ih:  IntrusivePtr<LinearPackedParamsBase>,

        /* Prepacked Weight Tensor */
        packed_w_hh:  IntrusivePtr<LinearPackedParamsBase>,

        /* float Bias Tensor */
        b_ih:         Tensor,

        /* float Bias Tensor */
        b_hh:         Tensor,

        /* Use reduced range for activation tensors */
        reduce_range: bool) -> Self {

        let reduce_range: bool = reduce_range.unwrap_or(false );

        todo!();
        /*


            : packed_w_ih(move(_packed_w_ih)),
            packed_w_hh(move(_packed_w_hh)),
            b_ih_(move(_b_ih)),
            b_hh_(move(_b_hh)),
            reduce_range_(_reduce_range)
        */
    }
    
    pub fn matmul_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(false, "matmul is not supported with quantized cell params");
        */
    }
    
    pub fn matmul_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(false, "matmul is not supported with quantized cell params");
        */
    }
    
    pub fn linear_ih(&self, input_ih: &Tensor) -> Tensor {
        
        todo!();
        /*
            return packed_w_ih->apply_dynamic(input_ih, reduce_range_);
        */
    }
    
    pub fn linear_hh(&self, input_hh: &Tensor) -> Tensor {
        
        todo!();
        /*
            return packed_w_hh->apply_dynamic(input_hh, reduce_range_);
        */
    }
    
    pub fn b_ih(&self) -> &Tensor {
        
        todo!();
        /*
            return b_ih_;
        */
    }
    
    pub fn b_hh(&self) -> &Tensor {
        
        todo!();
        /*
            return b_hh_;
        */
    }
    
    pub fn getstate(&self) -> CellParamsSerializationType {
        
        todo!();
        /*
            // Boxed dispatch nonsense
        // This will be cleaned up in the subsequent PR
        auto unpacked_ih = packed_w_ih->unpack();
        auto unpacked_hh = packed_w_hh->unpack();

        vector<Tensor> tensors_to_serialize{
            /*b_ih=*/b_ih_,
            /*b_hh=*/b_hh_,
        };

        vector<intrusive_ptr<LinearPackedParamsBase>>
            packed_params_to_serialize{packed_w_ih, packed_w_hh};

        // reduce_range parameter is serialized along with the int field values.
        return CellParamsSerializationType(
            "quantized_dynamic",
            move(tensors_to_serialize),
            {},
            {reduce_range_},
            move(packed_params_to_serialize));
        */
    }
    
    pub fn setstate(state: CellParamsSerializationType) -> IntrusivePtr<CellParamsBase> {
        
        todo!();
        /*
            vector<Tensor> tensors;
        vector<intrusive_ptr<LinearPackedParamsBase>> packed_params;
        vector<i64> serialized_ints;
        tie(ignore, tensors, ignore, serialized_ints, packed_params) =
            move(state);
        TORCH_INTERNAL_ASSERT(tensors.size() == 2);
        TORCH_INTERNAL_ASSERT(packed_params.size() == 2);

        bool reduce_range = serialized_ints.empty() ? false : serialized_ints[0];
        return make_quantized_cell_params_dynamic(
            /*w_ih_packed=*/move(packed_params[0]),
            /*w_hh_packed=*/move(packed_params[1]),
            /*bias_ih=*/move(tensors[0]),
            /*bias_hh=*/move(tensors[1]),
            /*reduce_range=*/reduce_range);
        */
    }
}

/**
  | QuantizedCellParams
  | vs. QuantizedCellParamsDynamic
  |
  | QuantizedCellParams uses the legacy
  | fbgemm_linear_int8_weight_fp32_activation API,
  | which requires the explicit scale and zero
  | point parameters for the weight.
  |
  | QuantizedCellParamsDynamic uses the new
  | fbgemm_linear_dynamic API, which doesn't
  | require the explicit scale and zero point
  | parameters.
  |
  | These quantization parameters are encapsulated
  | in the `PackedLinearWeight` struct in
  | aten/src/ATen/native/quantized/cpu/fbgemm_utils.h.
  */
pub fn make_quantized_cell_params_dynamic(
        w_ih_packed:  IntrusivePtr<LinearPackedParamsBase>,
        w_hh_packed:  IntrusivePtr<LinearPackedParamsBase>,
        bias_ih:      Tensor,
        bias_hh:      Tensor,
        reduce_range: bool) -> IntrusivePtr<CellParamsBase> {
    
    todo!();
        /*
            return make_intrusive<QuantizedCellParamsDynamic>(
          /*_packed_w_ih=*/move(w_ih_packed),
          /*_packed_w_hh=*/move(w_hh_packed),
          /*_b_ih=*/move(bias_ih),
          /*_b_hh=*/move(bias_hh),
          /*_reduce_range=*/reduce_range);
        */
}

pub struct QuantizedCellParamsFP16 {
    base:      CellParamsBase,
    packed_ih: IntrusivePtr<LinearPackedParamsBase>,
    packed_hh: IntrusivePtr<LinearPackedParamsBase>,
    b_ih:      Tensor,
    b_hh:      Tensor,
}

impl QuantizedCellParamsFP16 {
    
    pub fn new(
        packed_ih: IntrusivePtr<LinearPackedParamsBase>,
        packed_hh: IntrusivePtr<LinearPackedParamsBase>) -> Self {
    
        todo!();
        /*
        : packed_ih(move(_packed_ih)),
        : packed_hh(move(_packed_hh)),
        */
    }
    
    pub fn matmul_ih(&self, unused: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(false, "matmul is not supported with quantized cell params");
        */
    }
    
    pub fn matmul_hh(&self, unused: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(false, "matmul is not supported with quantized cell params");
        */
    }
    
    pub fn linear_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return packed_ih->apply_dynamic(input);
        */
    }
    
    pub fn linear_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return packed_hh->apply_dynamic(h);
        */
    }
    
    pub fn b_ih(&self) -> &Tensor {
        
        todo!();
        /*
            return b_ih_;
        */
    }
    
    pub fn b_hh(&self) -> &Tensor {
        
        todo!();
        /*
            return b_hh_;
        */
    }
    
    pub fn getstate(&self) -> CellParamsSerializationType {
        
        todo!();
        /*
            vector<intrusive_ptr<LinearPackedParamsBase>>
            packed_params_to_serialize{packed_ih, packed_hh};

        return CellParamsSerializationType(
            "quantized_fp16", {}, {}, {}, move(packed_params_to_serialize));
        */
    }
    
    pub fn setstate(state: CellParamsSerializationType) -> IntrusivePtr<CellParamsBase> {
        
        todo!();
        /*
            vector<intrusive_ptr<LinearPackedParamsBase>> packed_params;
        tie(
            ignore, ignore, ignore, ignore, packed_params) =
            move(state);
        TORCH_INTERNAL_ASSERT(packed_params.size() == 2);
        return make_quantized_cell_params_fp16(
            /*w_ih_packed=*/move(packed_params[0]),
            /*w_hh_packed=*/move(packed_params[1]));
        */
    }
}


pub fn make_quantized_cell_params_fp16(
        w_ih_packed: IntrusivePtr<LinearPackedParamsBase>,
        w_hh_packed: IntrusivePtr<LinearPackedParamsBase>) -> IntrusivePtr<CellParamsBase> {
    
    todo!();
        /*
            return make_intrusive<QuantizedCellParamsFP16>(
          move(w_ih_packed), move(w_hh_packed));
        */
}

lazy_static!{
    /*
    static unordered_map<
        string,
        intrusive_ptr<CellParamsBase> (*)(CellParamsSerializationType)>
        cell_params_deserializers = {
            {"quantized", &QuantizedCellParams::__setstate__},
            {"quantized_dynamic", &QuantizedCellParamsDynamic::__setstate__},
            {"quantized_fp16", &QuantizedCellParamsFP16::__setstate__}};
    */
}

/**
  | Stupid wrapper to convert from -> to
  | .
  |
  */
pub struct QRNNCellParamsWrapper {
    param: IntrusivePtr<CellParamsBase>,
}

impl QRNNCellParamsWrapper {
    
    pub fn new(param: IntrusivePtr<CellParamsBase>) -> Self {
    
        todo!();
        /*
            : param_(move(param))
        */
    }
    
    pub fn matmul_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return param_->matmul_ih(input);
        */
    }
    
    pub fn matmul_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return param_->matmul_hh(h);
        */
    }
    
    pub fn matmul_hr(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return param_->matmul_hr(h);
        */
    }
    
    pub fn linear_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return param_->linear_ih(input);
        */
    }
    
    pub fn linear_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return param_->linear_hh(h);
        */
    }
    
    pub fn b_ih(&self) -> &Tensor {
        
        todo!();
        /*
            return param_->b_ih();
        */
    }
    
    pub fn b_hh(&self) -> &Tensor {
        
        todo!();
        /*
            return param_->b_hh();
        */
    }
}


/**
  | Gathers every two elements of a vector
  | in a vector of pairs
  |
  */
pub fn pair_vec<T>(vals: &Vec<T>) -> Vec<(T,T)> {

    todo!();
        /*
            TORCH_CHECK(vals.size() % 2 == 0, "Odd number of params or hiddens given to a bidirectional RNN");
      vector<pair_of<T>> result;
      result.reserve(vals.size() / 2);
      for (usize i = 0; i < vals.size(); i += 2) {
        result.emplace_back(vals[i], vals[i + 1]);
      }
      return result;
        */
}

/**
  | Flattens a vector of pairs
  |
  */
pub fn unpair_vec<T>(vals: Vec<(T,T)>) -> Vec<T> {

    todo!();
        /*
            vector<T> result;
      result.reserve(vals.size() * 2);
      for (usize i = 0; i < vals.size(); i++) {
        result.push_back(move(vals[i].first));
        result.push_back(move(vals[i].second));
      }
      return result;
        */
}

/**
  | Parses a flat list of parameter tensors
  | into a list of CellParams
  |
  */
pub fn gather_params(
    params:          TensorList,
    has_biases:      bool,
    has_projections: bool) -> Vec<CellParams> {

    let has_projections: bool = has_projections.unwrap_or(false);

    todo!();
        /*
            static Tensor undefined;
      vector<CellParams> result;
      if (has_biases) {
        if (has_projections) {
          TORCH_CHECK(params.size() % 5 == 0, "got an incorrect number of RNN parameters");
          for (usize i = 0; i < params.size(); i += 5) {
            result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], params[i + 4]);
          }
        } else {
          TORCH_CHECK(params.size() % 4 == 0, "got an incorrect number of RNN parameters");
          for (usize i = 0; i < params.size(); i += 4) {
            result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], undefined);
          }
        }
      } else {
        if (has_projections) {
          TORCH_CHECK(params.size() % 3 == 0, "got an incorrect number of RNN parameters");
          for (usize i = 0; i < params.size(); i += 3) {
            result.emplace_back(params[i], params[i + 1], undefined, undefined, params[i + 2]);
          }
        } else {
          TORCH_CHECK(params.size() % 2 == 0, "got an incorrect number of RNN parameters");
          for (usize i = 0; i < params.size(); i += 2) {
            result.emplace_back(params[i], params[i + 1], undefined, undefined, undefined);
          }
        }
      }
      return result;
        */
}

/**
  | These gather_* functions are kept solely
  | for the purposes of backward compatbility
  | in the legacy quantized_{lstm,gru}
  | APIs
  |
  */
pub fn gather_quantized_params(params: List<Tensor>) -> List<IntrusivePtr<CellParamsBase>> {
    
    todo!();
        /*
            static Tensor undefined;
      vector<intrusive_ptr<CellParamsBase>> result;
      TORCH_CHECK(params.size() % 12 == 0, "got an incorrect number of quantized RNN parameters");
      for (usize i = 0; i < params.size(); i += 12) {
        result.emplace_back(make_intrusive<QuantizedCellParams>(
            static_cast<Tensor>(params[i]),
            static_cast<Tensor>(params[i + 1]),
            static_cast<Tensor>(params[i + 2]),
            static_cast<Tensor>(params[i + 3]),
            static_cast<Tensor>(params[i + 4]),
            static_cast<Tensor>(params[i + 5]),
            static_cast<Tensor>(params[i + 6]),
            static_cast<Tensor>(params[i + 7]),
            static_cast<Tensor>(params[i + 8]).item(),
            static_cast<Tensor>(params[i + 9]).item(),
            static_cast<Tensor>(params[i + 10]).item(),
            static_cast<Tensor>(params[i + 11]).item()));
      }
      return List<intrusive_ptr<CellParamsBase>>(result);
        */
}

pub fn gather_quantized_params_dynamic(params: List<Tensor>) -> List<IntrusivePtr<CellParamsBase>> {
    
    todo!();
        /*
            static Tensor undefined;
      vector<intrusive_ptr<CellParamsBase>> result;
      for (usize i = 0; i < params.size(); i += 2) {
        auto packed_struct_ih =
            cpp_custom_type_hack::cast<intrusive_ptr<LinearPackedParamsBase>>(
                static_cast<Tensor>(params[i]));
        auto packed_struct_hh =
            cpp_custom_type_hack::cast<intrusive_ptr<LinearPackedParamsBase>>(
                static_cast<Tensor>(params[i + 1]));

        auto bias_ih = packed_struct_ih->bias().value_or(undefined);
        auto bias_hh = packed_struct_hh->bias().value_or(undefined);
        result.emplace_back(make_intrusive<QuantizedCellParamsDynamic>(
            move(packed_struct_ih),
            move(packed_struct_hh),
            move(bias_ih),
            move(bias_hh)));
      }
      return List<intrusive_ptr<CellParamsBase>>(result);
        */
}

pub fn gather_quantized_params_fp16(params: List<Tensor>) -> List<IntrusivePtr<CellParamsBase>> {
    
    todo!();
        /*
            static Tensor undefined;
      vector<intrusive_ptr<CellParamsBase>> result;
      TORCH_CHECK(params.size() % 4 == 0,
                  "incorrect number of quantized RNN parameters FP16");
      for (usize i = 0; i < params.size(); i += 4) {
        intrusive_ptr<LinearPackedParamsBase> packed_struct_ih =
            cpp_custom_type_hack::cast<intrusive_ptr<LinearPackedParamsBase>>(
                static_cast<Tensor>(params[i]));
        intrusive_ptr<LinearPackedParamsBase> packed_struct_hh =
            cpp_custom_type_hack::cast<intrusive_ptr<LinearPackedParamsBase>>(
                static_cast<Tensor>(params[i + 1]));

        // NB: we install the bias from the gathered parameters here because
        // in the "new world", the fp16 linear apply() method always expects
        // the bias to be present in the packed struct. In the "old world",
        // we called `fbgemm_linear_fp16_weight_fp32_activation`, which took
        // the bias explicitly and ignored the bias in the packed struct. To
        // reconcile serialized models that behavied in the old style, we
        // put the bias into the appropriate packed structures here.
        //
        // Hopefully we can remove this in the future when we eliminate
        // the old style altogether
        packed_struct_ih->set_bias(params[i + 2]);
        packed_struct_hh->set_bias(params[i + 3]);

        result.emplace_back(make_intrusive<QuantizedCellParamsFP16>(
            move(packed_struct_ih), move(packed_struct_hh)));
      }
      return List<intrusive_ptr<CellParamsBase>>(result);
        */
}

/**
  | HIDDEN STATE FUNCTIONS
  |
  | Functions implemented below are implemented as
  | templates based on hidden type, because they
  | need to work both with simple RNNs and GRU
  | (which use a single Tensor), as well as with
  | LSTM (or possibly more complicated
  | architectures in the future).
  |
  | Still, there are some operations that need to
  | be performed on the hidden states alone, and
  | for this purpose we provide an overloaded set
  | of functions below.
  */
pub fn hidden_as_output_a(t: &Tensor) -> Tensor {
    
    todo!();
        /*
            return t;
        */
}

pub fn hidden_as_output_b(t: &(Tensor,Tensor)) -> Tensor {
    
    todo!();
        /*
            return get<0>(t);
        */
}

pub fn project<const index: usize>(tuples: ArrayRef<(Tensor,Tensor)>) -> Vec<Tensor> {

    todo!();
        /*
            vector<Tensor> result;
      result.reserve(tuples.size());
      for (auto & t : tuples) {
        result.push_back(get<index>(t));
      }
      return result;
        */
}

pub fn hidden_concat_a(hiddens: &[Tensor]) -> Tensor {
    
    todo!();
        /*
            return cat(hiddens, 0);
        */
}

pub fn hidden_concat_b(hiddens: ArrayRef<(Tensor,Tensor)>) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return make_tuple(hidden_concat(project<0>(hiddens)), hidden_concat(project<1>(hiddens)));
        */
}

pub fn hidden_slice_a(
    t:     &Tensor,
    start: i64,
    end:   i64) -> Tensor {

    todo!();
        /*
            return t.narrow(0, start, end - start);
        */
}

pub fn hidden_slice_b(
    t:     &(Tensor,Tensor),
    start: i64,
    end:   i64) -> (Tensor,Tensor) {

    todo!();
        /*
            return make_tuple(hidden_slice(get<0>(t), start, end),
                             hidden_slice(get<1>(t), start, end));
        */
}

/**
  | CELL IMPLEMENTATIONS
  |
  | Cell is a basic component of an RNN,
  | representing a single application of the
  | recurrent function. You can think of it as
  | a function of signature
  |
  | (Tensor input, hidden_type hidden, CellParams)
  | -> hidden_type
  |
  | which means that it consumes an input tensor,
  | and updates the previous hidden state.
  |
  | It's a struct only because functional
  | programming in C++ is a pain, and it's easier
  | to pass around "vtable pointers" than actual
  | function pointers.
  */
pub fn check_rnn_cell_forward_input(
        input:      &Tensor,
        input_size: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
        input.size(1) == input_size,
        "input has inconsistent input_size: got ", input.size(1), " expected ", input_size);
        */
}

pub fn check_rnn_cell_forward_hidden(
    input:        &Tensor,
    hx:           &Tensor,
    hidden_size:  i64,
    hidden_label: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
        input.size(0) == hx.size(0),
        "Input batch size ", input.size(0), " doesn't match hidden", hidden_label, " batch size ", hx.size(0));

      TORCH_CHECK(
        hx.size(1) == hidden_size,
        "hidden", hidden_label, " has inconsistent hidden_size: got ", hx.size(1), ", expected ", hidden_size);
        */
}

pub trait CellInterface<HiddenType,CellParams>:
InvokeCell<HiddenType,CellParams> {}

pub trait InvokeCell<HiddenType,CellParams> {
    
    fn invoke_cell(&self, 
        input:             &Tensor,
        hidden:            &HiddenType,
        params:            &CellParams,
        pre_compute_input: bool) -> HiddenType;
}

pub struct SimpleCell<nonlinearity,cell_params> {

}

impl<NonLinearity,CellParams> Cell<Tensor,CellParams> for SimpleCell<NonLinearity,CellParams> {
    
    fn invoke_cell(&self, 
        input:             &Tensor,
        hidden:            &Tensor,
        params:            &CellParams,
        pre_compute_input: bool) -> Tensor {
        let pre_compute_input: bool = pre_compute_input.unwrap_or(false);

        todo!();
        /*
            return nonlinearity{}(params.linear_hh(hidden).add_(
            pre_compute_input ? input : params.linear_ih(input)));
        */
    }
}


// TODO: can use inplace ops?
lazy_static!{
    /*
    template <typename cell_params>
    struct LSTMCell : Cell<tuple<Tensor, Tensor>, cell_params> {

      using hidden_type = tuple<Tensor, Tensor>;

      hidden_type operator()(
          const Tensor& input,
          const hidden_type& hidden,
          const cell_params& params,
          bool pre_compute_input = false) const override {
        const auto& hx = get<0>(hidden);
        const auto& cx = get<1>(hidden);

        if (input.is_cuda()) {
          TORCH_CHECK(!pre_compute_input);
          auto igates = params.matmul_ih(input);
          auto hgates = params.matmul_hh(hx);
          auto result = _thnn_fused_lstm_cell(
              igates, hgates, cx, params.b_ih(), params.b_hh());
          // applying projections if w_hr is defined
          auto hy = params.matmul_hr(get<0>(result));
          // Slice off the workspace argument (it's needed only for AD).
          return make_tuple(move(hy), move(get<1>(result)));
        }

        const auto gates = params.linear_hh(hx).add_(
            pre_compute_input ? input : params.linear_ih(input));
        auto chunked_gates = gates.unsafe_chunk(4, 1);
        auto ingate = chunked_gates[0].sigmoid_();
        auto forgetgate = chunked_gates[1].sigmoid_();
        auto cellgate = chunked_gates[2].tanh_();
        auto outgate = chunked_gates[3].sigmoid_();
        auto cy = (forgetgate * cx).add_(ingate * cellgate);
        auto hy = outgate * cy.tanh();
        hy = params.matmul_hr(hy);
        return make_tuple(move(hy), move(cy));
      }

    };

    template <typename cell_params>
    struct GRUCell : Cell<Tensor, cell_params> {
      using hidden_type = Tensor;

      hidden_type operator()(
          const Tensor& input,
          const hidden_type& hidden,
          const cell_params& params,
          bool pre_compute_input = false) const override {
        if (input.is_cuda()) {
          TORCH_CHECK(!pre_compute_input);
          auto igates = params.matmul_ih(input);
          auto hgates = params.matmul_hh(hidden);
          auto result = _thnn_fused_gru_cell(
              igates, hgates, hidden, params.b_ih(), params.b_hh());
          // Slice off the workspace argument (it's needed only for AD).
          return move(get<0>(result));
        }
        const auto chunked_igates = pre_compute_input
            ? input.unsafe_chunk(3, 1)
            : params.linear_ih(input).unsafe_chunk(3, 1);
        auto chunked_hgates = params.linear_hh(hidden).unsafe_chunk(3, 1);
        const auto reset_gate =
            chunked_hgates[0].add_(chunked_igates[0]).sigmoid_();
        const auto input_gate =
            chunked_hgates[1].add_(chunked_igates[1]).sigmoid_();
        const auto new_gate =
            chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate)).tanh_();
        return (hidden - new_gate).mul_(input_gate).add_(new_gate);
      }
    };

    ////////////////////////////////////////////////////////////
    // LAYER IMPLEMENTATIONS
    //
    // Layers are scan-like higher-order functions, which take in cells, and
    // transform them to functions of signature
    //
    // (io_type input, hidden_type hidden, param_type params) -> (io_type, hidden_type)
    //
    // which can apply the cell over a sequence of inputs, and produce both a new set
    // of hidden states, as well as a concatenated output of each step.

    template<typename output_type, typename hidden_type>
    struct LayerOutput {
      output_type outputs;
      hidden_type final_hidden;
    };

    template<typename io_type, typename hidden_type, typename param_type>
    struct Layer {
      using output_type = LayerOutput<io_type, hidden_type>;

      virtual ~Layer() {} // This is really dumb, but enables projects with
                          // -Wnon-virtual-dtor to compile...
      virtual output_type operator()(
          const io_type& input,
          const hidden_type& input_hidden,
          const param_type& params) const = 0;
    };

    template<typename hidden_type, typename cell_params>
    struct FullLayer : Layer<Tensor, hidden_type, cell_params> {
      using output_type =
          typename Layer<Tensor, hidden_type, cell_params>::output_type;
      using unstacked_output_type = LayerOutput<vector<Tensor>, hidden_type>;

      FullLayer(Cell<hidden_type, cell_params>& cell)
        : cell_(cell) {};

      unstacked_output_type operator()(
          const vector<Tensor>& step_inputs,
          const hidden_type& input_hidden,
          const cell_params& params,
          bool pre_compute_input = false) const {
        vector<Tensor> step_outputs;
        auto hidden = input_hidden;
        for (const auto& input : step_inputs) {
          hidden = cell_(input, hidden, params, pre_compute_input);
          step_outputs.emplace_back(hidden_as_output(hidden));
        }
        return {step_outputs, hidden};
      }

      output_type operator()(
          const Tensor& inputs,
          const hidden_type& input_hidden,
          const cell_params& params) const override {
        if (inputs.device().is_cpu()) {
          const auto inputs_w = params.linear_ih(inputs);
          auto unstacked_output =
              (*this)(inputs_w.unbind(0), input_hidden, params, true);
          return {stack(unstacked_output.outputs, 0),
                  unstacked_output.final_hidden};
        }
        auto unstacked_output = (*this)(inputs.unbind(0), input_hidden, params);
        return {stack(unstacked_output.outputs, 0),
                unstacked_output.final_hidden};
      }

      Cell<hidden_type, cell_params>& cell_;
    };

    template <typename dir_hidden_type, typename cell_params>
    struct FullBidirectionalLayer
        : Layer<Tensor, pair_of<dir_hidden_type>, pair_of<cell_params>> {
      using hidden_type = pair_of<dir_hidden_type>;
      using param_type = pair_of<cell_params>;
      using output_type = typename Layer<Tensor, hidden_type, param_type>::output_type;

      FullBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
        : layer_(cell) {};

      output_type operator()(
          const Tensor& input,
          const hidden_type& input_hidden,
          const param_type& params) const override {
        vector<Tensor> step_inputs;
        if (input.device().is_cpu()) {
          auto input_w = params.first.linear_ih(input);
          step_inputs = input_w.unbind(0);
          auto fw_result = layer_(
              step_inputs, input_hidden.first, params.first, true);
          auto fw_output = stack(fw_result.outputs, 0);
          input_w = params.second.linear_ih(input);
          step_inputs = input_w.unbind(0);
          auto rev_step_inputs = reverse(move(step_inputs));
          auto rev_result =
              layer_(rev_step_inputs, input_hidden.second, params.second, true);
          reverse(rev_result.outputs.begin(), rev_result.outputs.end());
          auto rev_output = stack(rev_result.outputs, 0);
          return {cat({fw_output, rev_output}, fw_output.dim() - 1),
                  make_pair(fw_result.final_hidden, rev_result.final_hidden)};
        }

        step_inputs = input.unbind(0);
        auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
        auto fw_output = stack(fw_result.outputs, 0);
        auto rev_step_inputs = reverse(move(step_inputs));
        auto rev_result =
            layer_(rev_step_inputs, input_hidden.second, params.second);
        reverse(rev_result.outputs.begin(), rev_result.outputs.end());
        auto rev_output = stack(rev_result.outputs, 0);
        return {cat({fw_output, rev_output}, fw_output.dim() - 1),
                make_pair(fw_result.final_hidden, rev_result.final_hidden)};
      }

      vector<Tensor> reverse(vector<Tensor>&& x) const {
        reverse(x.begin(), x.end());
        return move(x);
      }

      FullLayer<dir_hidden_type, cell_params> layer_;
    };

    template<typename hidden_type, typename cell_params>
    struct PackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
      using output_type =
          typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

      PackedLayer(Cell<hidden_type, cell_params>& cell)
        : cell_(cell) {};

      output_type operator()(
          const PackedSequence& input,
          const hidden_type& input_hidden,
          const cell_params& params) const override {

        vector<Tensor> step_outputs;
        vector<hidden_type> hiddens;
        i64 input_offset = 0;
        i64 num_steps = input.batch_sizes.size(0);
        i64* batch_sizes = input.batch_sizes.data_ptr<i64>();
        i64 last_batch_size = batch_sizes[0];

        const Tensor* input_ptr = &input.data;
        bool pre_compute_input = false;
        Tensor input_w;
        if (input.data.device().is_cpu()) {
          input_w = params.linear_ih(input.data);
          input_ptr = &input_w;
          pre_compute_input = true;
        }

        // Batch sizes is a sequence of decreasing lengths, which are offsets
        // into a 1D list of inputs. At every step we slice out batch_size elements,
        // and possibly account for the decrease in the batch size since the last step,
        // which requires us to slice the hidden state (since some sequences
        // are completed now). The sliced parts are also saved, because we will need
        // to return a tensor of final hidden state.
        auto hidden = input_hidden;
        for (i64 i = 0; i < num_steps; ++i) {
          const i64 batch_size = batch_sizes[i];
          auto step_input = input_ptr->narrow(0, input_offset, batch_size);
          input_offset += batch_size;
          const i64 dec = last_batch_size - batch_size;
          if (dec > 0) {
            hiddens.emplace_back(
                hidden_slice(hidden, last_batch_size - dec, last_batch_size));
            hidden = hidden_slice(hidden, 0, last_batch_size - dec);
          }

          last_batch_size = batch_size;
          hidden = cell_(step_input, hidden, params, pre_compute_input);
          step_outputs.push_back(hidden_as_output(hidden));
        }
        hiddens.emplace_back(hidden);
        reverse(hiddens.begin(), hiddens.end());

        return {PackedSequence{cat(step_outputs, 0), input.batch_sizes},
                hidden_concat(hiddens)};
      }

      Cell<hidden_type, cell_params>& cell_;
    };

    template<typename hidden_type, typename cell_params>
    struct ReversedPackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
      using output_type =
          typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

      ReversedPackedLayer(Cell<hidden_type, cell_params>& cell)
        : cell_(cell) {};

      output_type operator()(
          const PackedSequence& input,
          const hidden_type& input_hidden,
          const cell_params& params) const override {
        vector<Tensor> step_outputs;
        i64 input_offset = input.data.size(0);
        i64 num_steps = input.batch_sizes.size(0);
        i64* batch_sizes = input.batch_sizes.data_ptr<i64>();
        i64 last_batch_size = batch_sizes[num_steps - 1];

        const Tensor* input_ptr = &input.data;
        bool pre_compute_input = false;
        Tensor input_w;
        if (input.data.device().is_cpu()) {
          input_w = params.linear_ih(input.data);
          input_ptr = &input_w;
          pre_compute_input = true;
        }

        // Here the situation is similar to that above, except we start out with
        // the smallest batch size (and a small set of hidden states we actually use),
        // and progressively expand the hidden states, as we move backwards over the
        // 1D list of inputs.
        auto hidden = hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1]);
        for (i64 i = num_steps - 1; i >= 0; --i) {
          const i64 batch_size = batch_sizes[i];
          const i64 inc = batch_size - last_batch_size;
          if (inc > 0) {
            hidden = hidden_concat(ArrayRef<hidden_type>{
                hidden, hidden_slice(input_hidden, last_batch_size, batch_size)});
          }
          auto step_input =
              input_ptr->narrow(0, input_offset - batch_size, batch_size);
          input_offset -= batch_size;
          last_batch_size = batch_size;
          hidden = cell_(step_input, hidden, params, pre_compute_input);
          step_outputs.emplace_back(hidden_as_output(hidden));
        }
        reverse(step_outputs.begin(), step_outputs.end());
        return {PackedSequence{cat(step_outputs, 0), input.batch_sizes},
                hidden};
      }

      Cell<hidden_type, cell_params>& cell_;
    };

    template <typename dir_hidden_type, typename cell_params>
    struct PackedBidirectionalLayer
        : Layer<PackedSequence, pair_of<dir_hidden_type>, pair_of<cell_params>> {
      using hidden_type = pair_of<dir_hidden_type>;
      using param_type = pair_of<cell_params>;
      using output_type =
          typename Layer<PackedSequence, hidden_type, param_type>::output_type;

      PackedBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
        : layer_(cell), rev_layer_(cell) {};

      output_type operator()(
          const PackedSequence& input,
          const hidden_type& input_hidden,
          const param_type& params) const override {
        auto fw_result = layer_(input, input_hidden.first, params.first);
        auto rev_result = rev_layer_(input, input_hidden.second, params.second);
        PackedSequence output{
            cat({fw_result.outputs.data, rev_result.outputs.data}, -1),
            input.batch_sizes};
        return {output,
                make_pair(fw_result.final_hidden, rev_result.final_hidden)};
      }

      PackedLayer<dir_hidden_type, cell_params> layer_;
      ReversedPackedLayer<dir_hidden_type, cell_params> rev_layer_;
    };
    */
}


/**
  | apply_layer_stack
  |
  | layers are convenient, but in reality we often
  | want to stack them. this little helper manages
  | slicing of all inputs and parameters, and
  | repeatedly feeds them into the given layer.
  |
  | returns the last layer's outputs, and a vector
  | of final hidden states produced at each level.
  */
pub fn dropout_a(
    input: &Tensor,
    p:     f64) -> Tensor {
    
    todo!();
        /*
            return dropout(input, p, /*train=*/true);
        */
}

pub fn dropout_b(
    input: &PackedSequence,
    p:     f64) -> PackedSequence {
    
    todo!();
        /*
            return {dropout(input.data, p, /*train=*/true), input.batch_sizes};
        */
}

pub fn apply_layer_stack<io_type, hidden_type, weight_type>(
    layer:      &Layer<IoType,HiddenType,WeightType>,
    input:      &IoType,
    hiddens:    &Vec<HiddenType>,
    weights:    &Vec<WeightType>,
    num_layers: i64,
    dropout_p:  f64,
    train:      bool) -> LayerOutput<IoType,Vec<HiddenType>> {

    todo!();
        /*
            TORCH_CHECK(num_layers == (i64)hiddens.size(), "Expected more hidden states in stacked_rnn");
      TORCH_CHECK(num_layers == (i64)weights.size(), "Expected more weights in stacked_rnn");

      auto layer_input = input;
      auto hidden_it = hiddens.begin();
      auto weight_it = weights.begin();
      vector<hidden_type> final_hiddens;
      for (i64 l = 0; l < num_layers; ++l) {
        auto layer_output = layer(layer_input, *(hidden_it++), *(weight_it++));
        final_hiddens.push_back(layer_output.final_hidden);
        layer_input = layer_output.outputs;

        if (dropout_p != 0 && train && l < num_layers - 1) {
          layer_input = dropout(layer_input, dropout_p);
        }
      }

      return {layer_input, final_hiddens};
        */
}

////////////////////////////////////////////////////////////
// HELPERS SIMPLIFYING DISPATCH TO FUNCTIONS ABOVE
////////////////////////////////////////////////////////////

pub fn rnn_impl<CellType, LayerT, BidirLayerT, cell_params, io_type>(
    input:         &IoType,
    params:        &Vec<CellParams>,
    hiddens:       &Vec<CellType_HiddenType>,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool) -> LayerOutput<IoType,Vec<CellType_HiddenType>> {

    todo!();
        /*
            using hidden_type = typename CellType::hidden_type;
      CellType cell;
      if (bidirectional) {
        using BidirLayer = BidirLayerT<hidden_type, cell_params>;
        auto bidir_result = apply_layer_stack(BidirLayer{cell}, input, pair_vec(hiddens), pair_vec(params), num_layers, dropout_p, train);
        return {bidir_result.outputs, unpair_vec(move(bidir_result.final_hidden))};
      } else {
        return apply_layer_stack(LayerT<hidden_type,cell_params>{cell}, input, hiddens, params, num_layers, dropout_p, train);
      }
        */
}

pub fn rnn_impl_with_concat<CellType, LayerT, BidirLayerT, cell_params, io_type>(
    input:         &IoType,
    params:        &Vec<CellParams>,
    hiddens:       &Vec<CellType::hidden_type>,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool) -> (IoType,Tensor) {

    todo!();
        /*
            auto result = _rnn_impl<CellType, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);
      return make_tuple(move(result.outputs), stack(result.final_hidden, 0));
        */
}

pub fn lstm_impl<LayerT, BidirLayerT, cell_params, io_type>(
    input:         &IoType,
    params:        &Vec<CellParams>,
    hx:            &Tensor,
    cx:            &Tensor,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool) -> (IoType,Tensor,Tensor) {

    todo!();
        /*
            // It's much more useful for us to work on lists of pairs of hx and cx for each layer, so we need
      // to transpose a pair of those tensors.
      auto layer_hx = hx.unbind(0);
      auto layer_cx = cx.unbind(0);
      i64 total_layers = layer_hx.size();
      vector<typename LSTMCell<cell_params>::hidden_type> hiddens;
      hiddens.reserve(total_layers);
      for (i64 i = 0; i < total_layers; ++i) {
        hiddens.emplace_back(move(layer_hx[i]), move(layer_cx[i]));
      }

      auto result = _rnn_impl<LSTMCell<cell_params>, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);

      // Now, we need to reverse the transposed we performed above.
      vector<Tensor> hy, cy;
      hy.reserve(total_layers); cy.reserve(total_layers);
      for (auto & hidden : result.final_hidden) {
        hy.push_back(move(get<0>(hidden)));
        cy.push_back(move(get<1>(hidden)));
      }

      return make_tuple(move(result.outputs), stack(hy, 0), stack(cy, 0));
        */
}

pub fn use_cudnn_rnn_flatten_weight() -> bool {
    
    todo!();
        /*
            return getCUDAHooks().compiledWithCuDNN();
        */
}

////////////////////////////////////////////////////////////
// PUBLIC FUNCTIONS
////////////////////////////////////////////////////////////

#[macro_export] macro_rules! one_hidden_rnn {
    ($NAME:ident, $CELL:ty) => {
        /*
        
          define_dispatch(NAME##_cudnn_stub);                                       
          define_dispatch(NAME##_miopen_stub);                                      
          define_dispatch(NAME##_packed_cudnn_stub);                                
          define_dispatch(NAME##_packed_miopen_stub);                               
          REGISTER_NO_CPU_DISPATCH(NAME##_cudnn_stub, rnn_fn);                      
          REGISTER_NO_CPU_DISPATCH(NAME##_miopen_stub, rnn_fn);                     
          REGISTER_NO_CPU_DISPATCH(NAME##_packed_cudnn_stub, rnn_packed_fn);        
          REGISTER_NO_CPU_DISPATCH(NAME##_packed_miopen_stub, rnn_packed_fn);       
                                                                                    
          tuple<Tensor, Tensor> NAME(                                          
              const Tensor& _input,                                                 
              const Tensor& hx,                                                     
              TensorList _params,                                                   
              bool has_biases,                                                      
              i64 num_layers,                                                   
              double dropout_p,                                                     
              bool train,                                                           
              bool bidirectional,                                                   
              bool batch_first) {                                                   
            if (cudnn_is_acceptable(_input)) {                                  
              Tensor output, hy;                                                    
              NAME##_cudnn_stub(                                                    
                  _input.device().type(),                                           
                  output,                                                           
                  hy,                                                               
                  _input,                                                           
                  hx,                                                               
                  _params,                                                          
                  has_biases,                                                       
                  num_layers,                                                       
                  dropout_p,                                                        
                  train,                                                            
                  bidirectional,                                                    
                  batch_first);                                                     
              return make_tuple(move(output), move(hy));             
            }                                                                       
            if (use_miopen(_input, dropout_p)) {                                    
              Tensor output, hy;                                                    
              NAME##_miopen_stub(                                                   
                  _input.device().type(),                                           
                  output,                                                           
                  hy,                                                               
                  _input,                                                           
                  hx,                                                               
                  _params,                                                          
                  has_biases,                                                       
                  num_layers,                                                       
                  dropout_p,                                                        
                  train,                                                            
                  bidirectional,                                                    
                  batch_first);                                                     
              return make_tuple(move(output), move(hy));             
            }                                                                       
            check_attributes(_input, _params, hx);                                  
            auto input = batch_first ? _input.transpose(0, 1) : _input;             
            auto params = gather_params(_params, has_biases);                       
            auto results =                                                          
                _rnn_impl_with_concat<CELL, FullLayer, FullBidirectionalLayer>(     
                    input,                                                          
                    params,                                                         
                    hx.unbind(0),                                                   
                    num_layers,                                                     
                    dropout_p,                                                      
                    train,                                                          
                    bidirectional);                                                 
            if (batch_first) {                                                      
              get<0>(results).transpose_(0, 1);                                
            }                                                                       
            return results;                                                         
          }                                                                         
                                                                                    
          tuple<Tensor, Tensor> NAME(                                          
              const Tensor& data,                                                   
              const Tensor& batch_sizes,                                            
              const Tensor& hx,                                                     
              TensorList _params,                                                   
              bool has_biases,                                                      
              i64 num_layers,                                                   
              double dropout_p,                                                     
              bool train,                                                           
              bool bidirectional) {                                                 
            if (cudnn_is_acceptable(data)) {                                    
              Tensor output, hy;                                                    
              NAME##_packed_cudnn_stub(                                             
                  data.device().type(),                                             
                  output,                                                           
                  hy,                                                               
                  data,                                                             
                  batch_sizes,                                                      
                  hx,                                                               
                  _params,                                                          
                  has_biases,                                                       
                  num_layers,                                                       
                  dropout_p,                                                        
                  train,                                                            
                  bidirectional);                                                   
              return make_tuple(move(output), move(hy));             
            }                                                                       
            if (use_miopen(data, dropout_p)) {                                      
              Tensor output, hy;                                                    
              NAME##_packed_miopen_stub(                                            
                  data.device().type(),                                             
                  output,                                                           
                  hy,                                                               
                  data,                                                             
                  batch_sizes,                                                      
                  hx,                                                               
                  _params,                                                          
                  has_biases,                                                       
                  num_layers,                                                       
                  dropout_p,                                                        
                  train,                                                            
                  bidirectional);                                                   
              return make_tuple(move(output), move(hy));             
            }                                                                       
            PackedSequence input{data, batch_sizes};                                
            auto params = gather_params(_params, has_biases);                       
            auto result =                                                           
                _rnn_impl_with_concat<CELL, PackedLayer, PackedBidirectionalLayer>( 
                    input,                                                          
                    params,                                                         
                    hx.unbind(0),                                                   
                    num_layers,                                                     
                    dropout_p,                                                      
                    train,                                                          
                    bidirectional);                                                 
            auto& packed_output = get<0>(result);                              
            return make_tuple(                                                 
                move(packed_output.data), move(get<1>(result)));     
          }
        #define ONE_HIDDEN_QRNN(NAME, CELL)                                         
          tuple<Tensor, Tensor> NAME##_input(                                  
              const Tensor& _input,                                                 
              const Tensor& hx,                                                     
              List<intrusive_ptr<CellParamsBase>> _params,                
              bool has_biases,                                                      
              i64 num_layers,                                                   
              double dropout_p,                                                     
              bool train,                                                           
              bool bidirectional,                                                   
              bool batch_first) {                                                   
            vector<QRNNCellParamsWrapper> params;                              
            for (intrusive_ptr<CellParamsBase> x : _params) {                  
              params.emplace_back(move(x));                                    
            }                                                                       
            auto input = batch_first ? _input.transpose(0, 1) : _input;             
            auto results =                                                          
                _rnn_impl_with_concat<CELL, FullLayer, FullBidirectionalLayer>(     
                    input,                                                          
                    params,                                                         
                    hx.unbind(0),                                                   
                    num_layers,                                                     
                    dropout_p,                                                      
                    train,                                                          
                    bidirectional);                                                 
            if (batch_first) {                                                      
              get<0>(results).transpose_(0, 1);                                
            }                                                                       
            return results;                                                         
          }                                                                         
                                                                                    
          tuple<Tensor, Tensor> NAME##_data(                                   
              const Tensor& data,                                                   
              const Tensor& batch_sizes,                                            
              const Tensor& hx,                                                     
              List<intrusive_ptr<CellParamsBase>> _params,                
              bool has_biases,                                                      
              i64 num_layers,                                                   
              double dropout_p,                                                     
              bool train,                                                           
              bool bidirectional) {                                                 
            vector<QRNNCellParamsWrapper> params;                              
            for (intrusive_ptr<CellParamsBase> x : _params) {                  
              params.emplace_back(move(x));                                    
            }                                                                       
            PackedSequence input{data, batch_sizes};                                
            auto result =                                                           
                _rnn_impl_with_concat<CELL, PackedLayer, PackedBidirectionalLayer>( 
                    input,                                                          
                    params,                                                         
                    hx.unbind(0),                                                   
                    num_layers,                                                     
                    dropout_p,                                                      
                    train,                                                          
                    bidirectional);                                                 
            auto& packed_output = get<0>(result);                              
            return make_tuple(                                                 
                move(packed_output.data), move(get<1>(result)));     
          }
        */
    }
}

one_hidden_rnn!{gru, GRUCell<CellParams>}

lazy_static!{
    /*
    one_hidden_qrnn!{quantized_gru, GRUCell<QRNNCellParamsWrapper>}
    */
}

/// BC wrappers for quantized_gru
///
pub fn quantized_gru_input_legacy(
        input:         &Tensor,
        hx:            &Tensor,
        params:        List<Tensor>,
        has_biases:    bool,
        num_layers:    i64,
        dropout_p:     f64,
        train:         bool,
        bidirectional: bool,
        batch_first:   bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.quantized_gru with List[Tensor] for parameters is "
          "deprecated and may be removed! Please re-export your model "
          "using the newer definitions in torch.jit.quantized");
      auto params = gather_quantized_params(move(_params));
      return quantized_gru_input(
          _input,
          hx,
          move(params),
          has_biases,
          num_layers,
          dropout_p,
          train,
          bidirectional,
          batch_first);
        */
}

pub fn quantized_gru_data_legacy(
        data:          &Tensor,
        batch_sizes:   &Tensor,
        hx:            &Tensor,
        params:        List<Tensor>,
        has_biases:    bool,
        num_layers:    i64,
        dropout_p:     f64,
        train:         bool,
        bidirectional: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.quantized_gru with List[Tensor] for parameters is "
          "deprecated and may be removed! Please re-export your model "
          "using the newer definitions in torch.jit.quantized");
      auto params = gather_quantized_params(move(_params));
      return quantized_gru_data(
        data,
        batch_sizes,
        hx,
        move(params),
        has_biases,
        num_layers,
        dropout_p,
        train,
        bidirectional);
        */
}

pub type TanfCellType = SimpleCell<TanhF,CellParams>;

one_hidden_rnn!{rnn_tanh, tanf_cell_type}  

pub type ReulCellType = SimpleCell<ReluF,CellParams>;

one_hidden_rnn!(rnn_relu, relu_cell_type);

define_dispatch!{lstm_cudnn_stub}
define_dispatch!{lstm_packed_cudnn_stub}
define_dispatch!{lstm_miopen_stub}
define_dispatch!{lstm_packed_miopen_stub}

register_no_cpu_dispatch!{lstm_cudnn_stub         , lstm_fn}
register_no_cpu_dispatch!{lstm_packed_cudnn_stub  , lstm_packed_fn}
register_no_cpu_dispatch!{lstm_miopen_stub        , lstm_fn}
register_no_cpu_dispatch!{lstm_packed_miopen_stub , lstm_packed_fn}

pub fn lstm_a(
    input:         &Tensor,
    hx:            TensorList,
    params:        TensorList,
    has_biases:    bool,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool,
    batch_first:   bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
      if (cudnn_is_acceptable(_input)) {
        Tensor output, hy, cy;
        lstm_cudnn_stub(_input.device().type(), output, hy, cy, _input, hx, _params, has_biases,
                num_layers, dropout_p, train, bidirectional, batch_first);
        return make_tuple(move(output), move(hy), move(cy));
      }
      // if cells are of different size, that means projections are used
      bool has_projections = (hx[0].size(2) != hx[1].size(2));
      if (use_miopen(_input, dropout_p)) {
        if (!has_projections) {
          Tensor output, hy, cy;
          lstm_miopen_stub(_input.device().type(), output, hy, cy, _input, hx, _params, has_biases,
                    num_layers, dropout_p, train, bidirectional, batch_first);
          return make_tuple(move(output), move(hy), move(cy));
        } else {
          TORCH_WARN_ONCE(
              "LSTM with projections is not supported with MIOpen. Using default implementation.");
        }
      }

      check_attributes(_input, _params, hx);
      auto input = batch_first ? _input.transpose(0, 1) : _input;
      auto params = gather_params(_params, has_biases, has_projections);
      auto results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
          input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
      if (batch_first) {
        get<0>(results) = get<0>(results).transpose(0, 1);
      }
      return results;
        */
}

pub fn lstm_b(
    data:          &Tensor,
    batch_sizes:   &Tensor,
    hx:            TensorList,
    params:        TensorList,
    has_biases:    bool,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool) -> (Tensor,Tensor,Tensor) {

    todo!();
    /*
       TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
      if (cudnn_is_acceptable(data)) {
        Tensor output, hy, cy;
        lstm_packed_cudnn_stub(data.device().type(), output, hy, cy, data, batch_sizes, hx,
                _params, has_biases, num_layers, dropout_p, train, bidirectional);
        return make_tuple(move(output), move(hy), move(cy));
      }
      // if cells are of different size, that means projections are used
      bool has_projections = (hx[0].size(2) != hx[1].size(2));
      if (use_miopen(data, dropout_p)) {
        if (!has_projections) {
          Tensor output, hy, cy;
          lstm_packed_miopen_stub(data.device().type(), output, hy, cy, data, batch_sizes, hx,
                  _params, has_biases, num_layers, dropout_p, train, bidirectional);
          return make_tuple(move(output), move(hy), move(cy));
        } else {
          TORCH_WARN_ONCE(
              "LSTM with projections is not supported with MIOpen. Using default implementation.");
        }
      }

      PackedSequence input { data, batch_sizes };
      auto params = gather_params(_params, has_biases, has_projections);
      auto result = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
          input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
      auto & packed_output = get<0>(result);
      return make_tuple(move(packed_output.data),
                             move(get<1>(result)),
                             move(get<2>(result)));
        */
}

pub fn lstm_cell(
    input:    &Tensor,
    hx:       TensorList,
    w_ih:     &Tensor,
    w_hh:     &Tensor,
    b_ih_opt: &Option<Tensor>,
    b_hh_opt: &Option<Tensor>) -> (Tensor,Tensor) {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> b_ih_maybe_owned = borrow_from_optional_tensor(b_ih_opt);
      const Tensor& b_ih = *b_ih_maybe_owned;
      const Tensor& b_hh = value_or_else(b_hh_opt, [] {return Tensor();});

      TORCH_CHECK(hx.size() == 2, "lstm_cell expects two hidden states");
      check_rnn_cell_forward_input(input, w_ih.size(1));
      auto hidden_size = w_hh.size(1);
      check_rnn_cell_forward_hidden(input, hx[0], hidden_size, 0);
      check_rnn_cell_forward_hidden(input, hx[1], hidden_size, 0);
      static Tensor undefined;
      return LSTMCell<CellParams>{}(input, make_tuple(hx[0], hx[1]), CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
        */
}

pub fn thnn_differentiable_lstm_cell_backward(
    grad_hy_opt:     &Option<Tensor>,
    grad_cy_opt:     &Option<Tensor>,
    input_gates:     &Tensor,
    hidden_gates:    &Tensor,
    input_bias_opt:  &Option<Tensor>,
    hidden_bias_opt: &Option<Tensor>,
    cx:              &Tensor,
    cy:              &Tensor) -> (Tensor,Tensor,Tensor,Tensor,Tensor) {

    todo!();
    /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> grad_hy_maybe_owned = borrow_from_optional_tensor(grad_hy_opt);
      const Tensor& grad_hy = *grad_hy_maybe_owned;
      const Tensor& grad_cy = value_or_else(grad_cy_opt, [] {return Tensor();});
      const Tensor& input_bias = value_or_else(input_bias_opt, [] {return Tensor();});
      const Tensor& hidden_bias = value_or_else(hidden_bias_opt, [] {return Tensor();});

      if (!grad_hy.defined() && !grad_cy.defined()) {
        return tuple<Tensor, Tensor, Tensor, Tensor, Tensor>();
      }
      Tensor gates = input_gates + hidden_gates;
      if (input_bias.defined()) {
        gates = gates + input_bias;
      }
      if (hidden_bias.defined()) {
        gates = gates + hidden_bias;
      }
      auto chunked_gates = gates.unsafe_chunk(4, 1);
      Tensor i = chunked_gates[0].sigmoid();
      Tensor f = chunked_gates[1].sigmoid();
      Tensor c = chunked_gates[2].tanh();
      Tensor o = chunked_gates[3].sigmoid();

      Tensor gcx = cy.tanh();
      Tensor gog;
      TORCH_INTERNAL_ASSERT((grad_hy.defined() || grad_cy.defined()),"either gradient with respect to hy or cy should be defined");
      if (grad_hy.defined()) {
        gog = grad_hy * gcx;
        gog = sigmoid_backward(gog, o);
        gcx = tanh_backward(grad_hy * o, gcx);
        if (grad_cy.defined()) {
          gcx = gcx + grad_cy;
        }
      } else if (grad_cy.defined()) {
        gog = zeros_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        gcx = grad_cy;
      }
      Tensor gig = gcx * c;
      Tensor gfg = gcx * cx;
      Tensor gcg = gcx * i;
      gcx = gcx * f;
      gig = sigmoid_backward(gig, i);
      gfg = sigmoid_backward(gfg, f);
      gcg = tanh_backward(gcg, c);
      Tensor grad_gates = cat({gig, gfg, gcg, gog}, 1);
      Tensor grad_bias = input_bias.defined() ? grad_gates.sum(0, /*keepdim=*/false) : Tensor{};
      return make_tuple(grad_gates, grad_gates, move(gcx), grad_bias, grad_bias);
        */
}

pub fn thnn_differentiable_gru_cell_backward(
    grad_hy:         &Tensor,
    input_gates:     &Tensor,
    hidden_gates:    &Tensor,
    hx:              &Tensor,
    input_bias_opt:  &Option<Tensor>,
    hidden_bias_opt: &Option<Tensor>) -> (Tensor,Tensor,Tensor,Tensor,Tensor) {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> input_bias_maybe_owned = borrow_from_optional_tensor(input_bias_opt);
      const Tensor& input_bias = *input_bias_maybe_owned;
      const Tensor& hidden_bias = value_or_else(hidden_bias_opt, [] {return Tensor();});

      Tensor in_g = input_gates;
      Tensor h_g = hidden_gates;
      if (input_bias.defined()){
        in_g = in_g+input_bias;
      }
      if (hidden_bias.defined()){
        h_g = h_g + hidden_bias;
      }
      auto chunked_input_gates = in_g.unsafe_chunk(3, 1);
      Tensor ir = chunked_input_gates[0];
      Tensor ii = chunked_input_gates[1];
      Tensor in = chunked_input_gates[2];
      auto chunked_hidden_gates = h_g.unsafe_chunk(3, 1);
      Tensor hr = chunked_hidden_gates[0];
      Tensor hi = chunked_hidden_gates[1];
      Tensor hn = chunked_hidden_gates[2];
      Tensor rg = (ir + hr).sigmoid();
      Tensor ig = (ii + hi).sigmoid();
      Tensor grad_hx = grad_hy * ig;
      Tensor ng = (in+rg*hn).tanh();
      Tensor gig = sigmoid_backward(grad_hy * (hx - ng), ig);
      Tensor gin = tanh_backward(grad_hy * (1 - ig), ng);
      Tensor ghn = gin * rg;
      Tensor grg = sigmoid_backward(gin * hn, rg);
      Tensor grad_input_gates = cat({grg,gig,gin}, 1);
      Tensor grad_hidden_gates = cat({grg,gig,ghn}, 1);
      Tensor grad_input_bias = input_bias.defined() ? grad_input_gates.sum(0, /*keepdim=*/false) : Tensor{};
      Tensor grad_hidden_bias = input_bias.defined() ? grad_hidden_gates.sum(0, /*keepdim=*/false) : Tensor{};
      return make_tuple(move(grad_input_gates), move(grad_hidden_gates),
                             move(grad_hx), move(grad_input_bias), move(grad_hidden_bias));
        */
}

pub fn gru_cell(
    input:    &Tensor,
    hx:       &Tensor,
    w_ih:     &Tensor,
    w_hh:     &Tensor,
    b_ih_opt: &Option<Tensor>,
    b_hh_opt: &Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> b_ih_maybe_owned = borrow_from_optional_tensor(b_ih_opt);
      const Tensor& b_ih = *b_ih_maybe_owned;
      const Tensor& b_hh = value_or_else(b_hh_opt, [] {return Tensor();});

      check_rnn_cell_forward_input(input, w_ih.size(1));
      check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
      static Tensor undefined;
      return GRUCell<CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
        */
}

pub fn rnn_tanh_cell(
    input:    &Tensor,
    hx:       &Tensor,
    w_ih:     &Tensor,
    w_hh:     &Tensor,
    b_ih_opt: &Option<Tensor>,
    b_hh_opt: &Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> b_ih_maybe_owned = borrow_from_optional_tensor(b_ih_opt);
      const Tensor& b_ih = *b_ih_maybe_owned;
      const Tensor& b_hh = value_or_else(b_hh_opt, [] {return Tensor();});

      static Tensor undefined;
      check_rnn_cell_forward_input(input, w_ih.size(1));
      check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
      return SimpleCell<tanh_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
        */
}

pub fn rnn_relu_cell(
    input:    &Tensor,
    hx:       &Tensor,
    w_ih:     &Tensor,
    w_hh:     &Tensor,
    b_ih_opt: &Option<Tensor>,
    b_hh_opt: &Option<Tensor>) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> b_ih_maybe_owned = borrow_from_optional_tensor(b_ih_opt);
      const Tensor& b_ih = *b_ih_maybe_owned;
      const Tensor& b_hh = value_or_else(b_hh_opt, [] {return Tensor();});

      static Tensor undefined;
      check_rnn_cell_forward_input(input, w_ih.size(1));
      check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
      return SimpleCell<relu_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
        */
}

/**
  | Quantized implementations
  |
  | These implementations use FBGEMM to do the i2h
  | and h2h linear layers with an int8 or float16
  | quantized weight. This is advantageous in
  | small-batch-size scenarios where runtime is
  | dominated by memory fetches of the weight
  | matrix.
  */
pub fn quantized_lstm_input(
    input:         &Tensor,
    hx:            List<Tensor>,
    params:        List<IntrusivePtr<CellParamsBase>>,
    has_biases:    bool,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool,
    batch_first:   bool,
    dtype:         Option<ScalarType>,
    use_dynamic:   bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            auto hx = hx_.vec();
      vector<QRNNCellParamsWrapper> params;
      params.reserve(_params_.size());
      for (const auto& param : _params_) {
        params.emplace_back(static_cast<intrusive_ptr<CellParamsBase>>(param));
      }
      TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
      TORCH_CHECK(hx[0].size(2) == hx[1].size(2), "quantized LSTM with projections is not supported");
      auto result_dtype = dtype.has_value() ? dtype.value() : kChar;
      auto input = batch_first ? _input.transpose(0, 1) : _input;
      TORCH_CHECK(has_biases, "quantized LSTM requires biases");
      TORCH_CHECK(
          result_dtype == kChar || result_dtype == kQInt8 ||
              result_dtype == kHalf,
          "dtype is not supported");

      tuple<Tensor, Tensor, Tensor> results;
      if (result_dtype == kChar || result_dtype == kQInt8) {
        if (use_dynamic) {
          results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
              input, params, hx[0], hx[1], num_layers,
              dropout_p, train, bidirectional);
        } else {
          results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
              input, params, hx[0], hx[1], num_layers,
              dropout_p, train, bidirectional);
        }
      } else {
        results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
            input, params, hx[0], hx[1], num_layers,
            dropout_p, train, bidirectional);
      }

      if (batch_first) {
        get<0>(results) = get<0>(results).transpose(0, 1);
      }
      return results;
        */
}

/**
  | BC wrappers for quantized_lstm
  |
  */
pub fn quantized_lstm_input_legacy(
    input:         &Tensor,
    hx:            List<Tensor>,
    params:        List<Tensor>,
    has_biases:    bool,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool,
    batch_first:   bool,
    dtype:         Option<ScalarType>,
    use_dynamic:   bool) -> (Tensor,Tensor,Tensor) {

    todo!();
    /*
       TORCH_WARN_ONCE(
       "torch.quantized_lstm with List[Tensor] for parameters is "
          "deprecated and may be removed! Please re-export your model "
          "using the newer definitions in torch.jit.quantized");
      List<intrusive_ptr<CellParamsBase>> params;
      auto result_dtype = dtype.has_value() ? dtype.value() : kChar;
      if (result_dtype == kChar || result_dtype == kQInt8) {
        if (use_dynamic) {
          params = gather_quantized_params_dynamic(move(_params_));
        } else {
          params = gather_quantized_params(move(_params_));
        }
      } else {
        params = gather_quantized_params_fp16(move(_params_));
      }
      return quantized_lstm_input(
          _input,
          move(hx_),
          move(params),
          has_biases,
          num_layers,
          dropout_p,
          train,
          bidirectional,
          batch_first,
          move(dtype),
          use_dynamic);
        */
}

pub fn quantized_lstm_data(
    data:          &Tensor,
    batch_sizes:   &Tensor,
    hx:            List<Tensor>,
    params:        List<IntrusivePtr<CellParamsBase>>,
    has_biases:    bool,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool,
    dtype:         Option<ScalarType>,
    use_dynamic:   bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            auto hx = hx_.vec();
      vector<QRNNCellParamsWrapper> params;
      params.reserve(_params_.size());
      for (const auto& param : _params_) {
        params.emplace_back(static_cast<intrusive_ptr<CellParamsBase>>(param));
      }
      TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
      TORCH_CHECK(hx[0].size(2) == hx[1].size(2), "quantized LSTM with projections is not supported");

      auto result_dtype = dtype.has_value() ? dtype.value() : kChar;

      PackedSequence input { data, batch_sizes };
      tuple<PackedSequence, Tensor, Tensor> results;
      if (result_dtype == kChar || result_dtype == kQInt8) {
        if (use_dynamic) {
          results = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
              input, params, hx[0], hx[1], num_layers,
              dropout_p, train, bidirectional);
        } else {
          results = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
              input, params, hx[0], hx[1], num_layers,
              dropout_p, train, bidirectional);
        }
      } else {
        results = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
            input, params, hx[0], hx[1], num_layers,
            dropout_p, train, bidirectional);
      }
      auto & packed_output = get<0>(results);
      return make_tuple(move(packed_output.data),
                             move(get<1>(results)),
                             move(get<2>(results)));
        */
}

pub fn quantized_lstm_data_legacy(
    data:          &Tensor,
    batch_sizes:   &Tensor,
    hx:            List<Tensor>,
    params:        List<Tensor>,
    has_biases:    bool,
    num_layers:    i64,
    dropout_p:     f64,
    train:         bool,
    bidirectional: bool,
    dtype:         Option<ScalarType>,
    use_dynamic:   bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.quantized_lstm with List[Tensor] for parameters is "
          "deprecated and may be removed! Please re-export your model "
          "using the newer definitions in torch.jit.quantized");
      List<intrusive_ptr<CellParamsBase>> params;
      auto result_dtype = dtype.has_value() ? dtype.value() : kChar;
      if (result_dtype == kChar || result_dtype == kQInt8) {
        if (use_dynamic) {
          params = gather_quantized_params_dynamic(move(_params_));
        } else {
          params = gather_quantized_params(move(_params_));
        }
      } else {
        params = gather_quantized_params_fp16(move(_params_));
      }
      return quantized_lstm_data(
          data,
          batch_sizes,
          move(hx_),
          move(params),
          has_biases,
          num_layers,
          dropout_p,
          train,
          bidirectional,
          move(dtype),
          use_dynamic);
        */
}

#[macro_export] macro_rules! define_quantized_rnn_cell {
    ($name:ident, $hx_type:ident, $cell_type:ident, $return_type:ident, $prepare_hx_fn:ident) => {
        /*
        
        return_type name( 
            const Tensor& input, 
            hx_type hx, 
            const Tensor& w_ih, 
            const Tensor& w_hh, 
            const Tensor& b_ih, 
            const Tensor& b_hh, 
            const Tensor& packed_ih, 
            const Tensor& packed_hh, 
            const Tensor& col_offsets_ih, 
            const Tensor& col_offsets_hh, 
            const Scalar& scale_ih, 
            const Scalar& scale_hh, 
            const Scalar& zero_point_ih, 
            const Scalar& zero_point_hh) { 
          QuantizedCellParams params( 
              w_ih, 
              w_hh, 
              b_ih, 
              b_hh, 
              packed_ih, 
              packed_hh, 
              col_offsets_ih, 
              col_offsets_hh, 
              scale_ih, 
              scale_hh, 
              zero_point_ih, 
              zero_point_hh); 
          return cell_type{}( 
              input, prepare_hx_fn(hx), params); 
        }
        */
    }
}

/**
  | Set reduced range to be True for all RNN Cells
  | by default. This flag is used only for FBGEMM
  | kernels QNNPACK does not reduce range for
  | activations
  |
  */
#[macro_export] macro_rules! define_quantized_rnn_cell_dynamic {
    ($name:ident, $hx_type:ident, $cell_type:ident, $return_type:ident, $prepare_hx_fn:ident) => {
        /*
        
        return_type name( 
            const Tensor& input, 
            hx_type hx, 
            intrusive_ptr<LinearPackedParamsBase> _packed_w_ih, 
            intrusive_ptr<LinearPackedParamsBase> _packed_w_hh, 
            const Tensor& b_ih, 
            const Tensor& b_hh 
         ) { 
          QuantizedCellParamsDynamic params( 
              _packed_w_ih, 
              _packed_w_hh, 
              b_ih, 
              b_hh,
              true); 
          return cell_type{}( 
              input, prepare_hx_fn(hx), params); 
        }
        */
    }
}

// Quantized LSTM cell
pub type QuantizedLstmCellType   = LSTMCell<QuantizedCellParams>;
pub type QuantizedLstmReturnType = (Tensor,Tensor);

pub fn prepare_quantized_lstm_hx(hx: TensorList) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return make_tuple(hx[0], hx[1]);
        */
}

// Quantized LSTM cell
pub type QuantizedLstmCellDynamicType = LSTMCell<QuantizedCellParamsDynamic>;

define_quantized_rnn_cell!(
    quantized_lstm_cell, 
    TensorList, 
    quantized_lstm_cell_type, 
    quantized_lstm_return_type, 
    prepare_quantized_lstm_hx
);

define_quantized_rnn_cell_dynamic!(
    quantized_lstm_cell_dynamic, 
    TensorList, 
    quantized_lstm_cell_dynamic_type, 
    quantized_lstm_return_type, 
    prepare_quantized_lstm_hx
);

// Helpers for simpler cells
pub type SimpleHxType = Arc<Tensor>;

pub fn prepare_quantized_hx(hx: SimpleHxType) -> SimpleHxType {
    
    todo!();
        /*
            return hx;
        */
}

// Quantized GRU cell
pub type QuantizedGruCellType        = GRUCell<QuantizedCellParams>;
pub type QuantizedGruCellDynamicType = GRUCell<QuantizedCellParamsDynamic>;

define_quantized_rnn_cell!(
    quantized_gru_cell, 
    simple_hx_type, 
    quantized_gru_cell_type, 
    Tensor, 
    prepare_quantized_hx
);

define_quantized_rnn_cell_dynamic!(
    quantized_gru_cell_dynamic, 
    simple_hx_type, 
    quantized_gru_cell_dynamic_type, 
    Tensor, 
    prepare_quantized_hx
);

// Quantized RNN w/ ReLU cell
pub type QuantizedRnnReluCellType = SimpleCell<ReluF,QuantizedCellParams>;

define_quantized_rnn_cell!(
    quantized_rnn_relu_cell, 
    simple_hx_type, 
    quantized_rnn_relu_cell_type, 
    Tensor, 
    prepare_quantized_hx
);

pub type QuantizedRnnReluCellDynamicType = SimpleCell<ReluF,QuantizedCellParamsDynamic>;

define_quantized_rnn_cell_dynamic!(
    quantized_rnn_relu_cell_dynamic, 
    simple_hx_type, 
    quantized_rnn_relu_cell_dynamic_type, 
    Tensor, 
    prepare_quantized_hx
);

// Quantized RNN w/ tanh cell
pub type QuantizedRnnTanhCellType = SimpleCell<TanhF,QuantizedCellParams>;

define_quantized_rnn_cell!(
    quantized_rnn_tanh_cell, 
    simple_hx_type, 
    quantized_rnn_tanh_cell_type, 
    Tensor, 
    prepare_quantized_hx
);

pub type QuantizedRnnTanhCellDynamicType = SimpleCell<TanhF,QuantizedCellParamsDynamic>;

define_quantized_rnn_cell_dynamic!(
    quantized_rnn_tanh_cell_dynamic, 
    simple_hx_type, 
    quantized_rnn_tanh_cell_dynamic_type, 
    Tensor, 
    prepare_quantized_hx
);

lazy_static!{
    /*
    static auto ensure_linear_params_registered = register_linear_params();
    */
}

lazy_static!{
    /*
    static auto cell_params_base_registry =
        Torchclass_<CellParamsBase>("rnn", "CellParamsBase")
            .def_pickle(
                [](const intrusive_ptr<CellParamsBase>& self)
                    -> CellParamsSerializationType { return self->__getstate__(); },
                [](CellParamsSerializationType state)
                    -> intrusive_ptr<CellParamsBase> {
                  string type = get<0>(state);
                  TORCH_INTERNAL_ASSERT(cell_params_deserializers.count(type));
                  return cell_params_deserializers[type](move(state));
                });
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_FRAGMENT(aten, m) {
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_lstm.input(Tensor input, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_gru.input(Tensor input, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"));
      m.def(
          TORCH_SELECTIVE_SCHEMA("quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_FRAGMENT(quantized, m) {
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::make_quantized_cell_params_dynamic(__torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh, bool reduce_range=False) -> __torch__.torch.classes.rnn.CellParamsBase"));
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::make_quantized_cell_params_fp16(__torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh) -> __torch__.torch.classes.rnn.CellParamsBase"));
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::make_quantized_cell_params(Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> __torch__.torch.classes.rnn.CellParamsBase"));
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_lstm_cell_dynamic(Tensor input, Tensor[] hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh) -> (Tensor, Tensor)"));
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_gru_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor"));
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_rnn_relu_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor"));
      m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_rnn_tanh_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor"));
    }

    TORCH_LIBRARY_IMPL(aten, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized_lstm.input"), TORCH_FN(quantized_lstm_input));
      m.impl(TORCH_SELECTIVE_NAME("quantized_lstm.data"), TORCH_FN(quantized_lstm_data));
      m.impl(TORCH_SELECTIVE_NAME("quantized_lstm.input_legacy"), TORCH_FN(quantized_lstm_input_legacy));
      m.impl(TORCH_SELECTIVE_NAME("quantized_lstm.data_legacy"), TORCH_FN(quantized_lstm_data_legacy));
      m.impl(TORCH_SELECTIVE_NAME("quantized_gru.input"), TORCH_FN(quantized_gru_input));
      m.impl(TORCH_SELECTIVE_NAME("quantized_gru.data"), TORCH_FN(quantized_gru_data));
      m.impl(TORCH_SELECTIVE_NAME("quantized_gru.input_legacy"), TORCH_FN(quantized_gru_input_legacy));
      m.impl(TORCH_SELECTIVE_NAME("quantized_gru.data_legacy"), TORCH_FN(quantized_gru_data_legacy));
    }

    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::make_quantized_cell_params_dynamic"), TORCH_FN(make_quantized_cell_params_dynamic));
      m.impl(TORCH_SELECTIVE_NAME("quantized::make_quantized_cell_params"), TORCH_FN(make_quantized_cell_params));
      m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_lstm_cell_dynamic"), TORCH_FN(quantized_lstm_cell_dynamic));
      m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_gru_cell_dynamic"), TORCH_FN(quantized_gru_cell_dynamic));
      m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_rnn_relu_cell_dynamic"), TORCH_FN(quantized_rnn_relu_cell_dynamic));
      m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_rnn_tanh_cell_dynamic"), TORCH_FN(quantized_rnn_tanh_cell_dynamic));
    }

    TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::make_quantized_cell_params_fp16"), TORCH_FN(make_quantized_cell_params_fp16));
    }
    */
}
