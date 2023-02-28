crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A templated class to allow one to wrap
  | a CPU operator as an IDEEP operator.
  | 
  | This class can be used when one does not
  | have the IDEEP implementation ready
  | yet for an operator.
  | 
  | Essentially, what this op does is to
  | automatically deal with data copy for
  | you. Plausibly, this causes a lot of
  | overhead and is not optimal, so you should
  | use this operator mostly for quick prototyping
  | purpose.
  | 
  | All the input and output of the original
  | operator should be TensorCPU.
  | 
  | Example usage: if you have a class MyMagicOp
  | that is CPU based, and you use the registration
  | code
  | 
  | REGISTER_CPU_OPERATOR(MyMagic,
  | MyMagicOp);
  | 
  | to register the CPU side, you can create
  | its corresponding IDEEP operator (with
  | performance hits of course) via
  | 
  | REGISTER_IDEEP_OPERATOR(MyMagic,
  | 
  | IDEEPFallbackOp<MyMagicOp>);
  | 
  | Advanced usage: if you want to have some
  | specific outputs never copied, you
  | can use the
  | 
  | SkipOutputCopy template argument
  | to do that.
  | 
  | For example, if MyMagic produces two
  | outputs and the first output is always
  | going to live on the
  | 
  | CPU, you can do
  | 
  | REGISTER_IDEEP_OPERATOR(MyMagic,
  | 
  | IDEEPFallbackOp<MyMagicOp, SkipIndices<0>>);
  |
  */
pub type SkipOutputCopyDefault<const V: i32> = dyn SkipIndices<V>;

pub struct IDEEPFallbackOp<CPUOp, SkipOutputCopy: ?Sized> {
    local_input_blobs:   Vec<*mut Blob>,
    local_output_blobs:  Vec<*mut Blob>,
    output_inplace:      Vec<bool>,
    input_share:         Vec<bool>,
    base_op:             Box<CPUOp>,
    local_ws:            Box<Workspace>,
    base_def:            OperatorDef,
    phantom:             PhantomData<SkipOutputCopy>,
}

impl Operator for IDEEPFallbackOp<CPUOp, SkipOutputCopy> {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
}

impl IDEEPFallbackOp<CPUOp, SkipOutputCopy> {
    
    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(def, ws) 

        CAFFE_ENFORCE_EQ(def.device_option().device_type(), PROTO_IDEEP);
        base_def_.CopyFrom(def);
        // base_def_ runs on CPU, so we will set its device option to CPU.
        // Copy to allow random_seed to be correctly propagated.
        base_def_.mutable_device_option()->CopyFrom(def.device_option());
        base_def_.mutable_device_option()->set_device_type(PROTO_CPU);
        // Create output blobs in parent workspace,
        // then forward output blobs to local workspace.
        std::unordered_map<string, string> forwarded_output_blobs;
        for (int i = 0; i < base_def_.output_size(); i++) {
          // For in-place case, the in/output tensor for local_ws must be
          // re-created, instead of forwarding from current workspace.
          string parent_name(base_def_.output(i));
          if (!SkipOutputCopy::Contains(i)) {
            parent_name += "_cpu_output_blob_" + base_def_.type();
          }
          local_output_blobs_.push_back(ws->CreateBlob(parent_name));
          CHECK_NOTNULL(local_output_blobs_.back());
          forwarded_output_blobs[base_def_.output(i)] = parent_name;
          output_inplace_.push_back(false);
          for (const string &input_name : base_def_.input()) {
            if (input_name == base_def_.output(i)) {
              output_inplace_[i] = true;
              break;
            }
          }
        }
        local_ws_.reset(new Workspace(ws, forwarded_output_blobs));
        // Set up the symbols for the local workspace.
        for (const string& name : base_def_.input()) {
          local_input_blobs_.push_back(local_ws_->CreateBlob(name));
          CHECK_NOTNULL(local_input_blobs_.back());
        }
        input_share_.resize(local_input_blobs_.size(), false);
        base_op_.reset(new CPUOp(base_def_, local_ws_.get()));
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            for (int i = 0; i < InputSize(); ++i) {
          if (InputIsType<itensor>(i)
              && (Input(i).has_scale()
                || Input(i).get_data_type() == idtype::f32)) {
            auto& input = Input(i);
            if (input_share_[i]) {
              local_input_blobs_[i]->Reset();
              input_share_[i] = false;
            }
            auto dtensor = BlobGetMutableTensor(local_input_blobs_[i], CPU);
            dtensor->Resize(input.get_dims());
            // If fallback from INT8, the public format of original input is nhwc.
            // While the required format is nchw, need to reorder to nchw.
            if (input.get_desc().is_nhwc()) {
              itensor temp_ten ({input.get_dims(), idtype::f32, iformat::nchw},
                  dtensor->template mutable_data<float>());
              temp_ten.feed_from(input);
            } else if (!input.need_reorder()) {
              CAFFE_ENFORCE(!input.has_scale(),
                  "Incorrect invocation of get_data_handle");
              dtensor->ShareExternalPointer(
                  static_cast<float*>(input.get_data_handle()));
            } else {
              input.to_public(dtensor->template mutable_data<float>());
            }
          } else {
            VLOG(1) << "Input " << i << " is not ideep::tensor. Skipping copy.";
            if (OperatorStorage::Inputs()[i]->GetRaw() != local_input_blobs_[i]->GetRaw()) {
              // Note(jiayq): This removes a const but conceptually
              // local_input_blobs will only be used as const blob input for the
              // base op so we are still fine.
              local_input_blobs_[i]->ShareExternal(
                  const_cast<void *>(OperatorStorage::Inputs()[i]->GetRaw()),
                  OperatorStorage::Inputs()[i]->meta());
            }
            input_share_[i] = true;
          }
        }

        // Some CPU ops inherited from OperatorStorage directly might need this default
        // input argument '0' like 'PrefetchOperator'.
        if (!base_op_->Run(0)) {
          LOG(ERROR) << "Base op run failed in IDEEPFallbackOp. Def: "
                     << ProtoDebugString(this->debug_def());
          return false;
        }

        for (int i = 0; i < OutputSize(); ++i) {
          if (SkipOutputCopy::Contains(i)) {
            VLOG(1) << "Copy output: index " << i << " skipped.";
            continue;
          }
          CAFFE_ENFORCE(
              BlobIsTensorType(*local_output_blobs_[i], CPU),
              "IDEEP fallback op currently does not support non-TensorCPU "
              "output type who needs copying.");
          const auto& src = local_output_blobs_[i]->template Get<TensorCPU>();
          auto src_dims = src.sizes().vec();
          if (src.template IsType<float>() && src.dim() != 0 && base_op_->type() != "Python") {
            Blob* dst = OperatorStorage::OutputBlob(i);
            // The output tensor must be ideep tensor with public format.
            // If reusing ideep tensor with non-public format, the tensor buffer
            // will be interpreted incorrectly.
            if (!dst->template IsType<itensor>() ||
                !dst->template Get<itensor>().is_public_format()) {
              dst->Reset(new itensor());
            }

            itensor::dims dst_dims (src_dims.begin(), src_dims.end());
            auto dtensor = dst->template GetMutable<itensor>();
            if (dtensor->get_dims() != dst_dims) {
              dtensor->resize(dst_dims, idtype::f32);
            }
            if (output_inplace_[i]) {
              dtensor->feed_from(dst_dims, idtype::f32,
                  const_cast<void*>(src.raw_data()));
            } else {
              CAFFE_ENFORCE(!dtensor->has_scale(),
                  "Incorrect invocation of set_data_handle");
              dtensor->set_data_handle(const_cast<void *>(src.raw_data()));
            }
          } else {
            VLOG(2) << "Output " << base_def_.output(i) << " as CPUTensor";
            Blob* dst = OperatorStorage::OutputBlob(i);
            if (output_inplace_[i]) {
              auto dtensor = BlobGetMutableTensor(dst, CPU);
              dtensor->CopyFrom(src);
            } else {
              dst->Reset(new Tensor(CPU));
              BlobSetTensor(dst, src.Alias());
            }
          }
        }
        return true;
        */
    }
}

// Boolean operators
register_ideep_compare_operator!{EQ}
register_ideep_compare_operator!{GT}
register_ideep_compare_operator!{GE}
register_ideep_compare_operator!{LT}
register_ideep_compare_operator!{LE}
register_ideep_compare_operator!{NE}

register_ideep_operator!{
    Softmax,
    IDEEPFallbackOp::<SoftmaxOp::<f32,CPUContext>>
}
register_ideep_operator!{
    LabelCrossEntropy,
    IDEEPFallbackOp::<LabelCrossEntropyOp::<f32,CPUContext>>
}
register_ideep_operator!{
    AveragedLoss,
    IDEEPFallbackOp::<AveragedLoss::<f32,CPUContext>,SkipIndices::<0>>
}
register_ideep_operator!{
    Flatten,
    IDEEPFallbackOp::<FlattenOp::<CPUContext>>
}

register_ideep_operator!{
    ResizeLike,
    IDEEPFallbackOp::<ResizeLikeOp::<CPUContext>>
}

register_ideep_operator!{
    Slice,
    IDEEPFallbackOp::<SliceOp::<CPUContext>>
}

register_ideep_operator!{
    Clip,
    IDEEPFallbackOp::<ClipOp::<f32,CPUContext>>
}

register_ideep_operator!{
    ScatterAssign,
    IDEEPFallbackOp::<ScatterAssignOp::<CPUContext>>
}

register_ideep_operator!{
    Cast,
    IDEEPFallbackOp::<CastOp::<CPUContext>>
}

// filter operators
register_ideep_operator!{
    XavierFill,
    IDEEPFallbackOp::<XavierFillOp<f32,CPUContext>>
}

register_ideep_operator!{
    ConstantFill,
    IDEEPFallbackOp::<ConstantFillOp::<CPUContext>>
}

register_ideep_operator!{
    GaussianFill,
    IDEEPFallbackOp::<GaussianFillOp::<f32,CPUContext>>
}

register_ideep_operator!{
    MSRAFill,
    IDEEPFallbackOp::<MSRAFillOp::<f32,CPUContext>>
}

register_ideep_operator!{
    GivenTensorFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<f32,CPUContext>>
}

// Not supported tensor types in below FillOp
register_ideep_operator!{
    GivenTensorDoubleFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<f64,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorBoolFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<bool,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorIntFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<i32,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorInt64Fill,
    IDEEPFallbackOp::<GivenTensorFillOp::<i64,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorStringFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<String,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    Load,
    IDEEPFallbackOp::<LoadOp::<CPUContext>>
}

register_ideep_operator!{
    Save,
    IDEEPFallbackOp::<SaveOp::<CPUContext>>
}

register_ideep_operator!{
    RMACRegions,
    IDEEPFallbackOp::<RMACRegionsOp::<CPUContext>>
}

register_ideep_operator!{
    RoIPool,
    IDEEPFallbackOp::<RoIPoolOp::<f32,CPUContext>>
}

register_ideep_operator!{
    RoIAlign,
    IDEEPFallbackOp::<RoIAlignOp::<f32,CPUContext>>
}

register_ideep_operator!{
    RoIAlignRotated,
    IDEEPFallbackOp::<RoIAlignRotatedOp::<f32,CPUContext>>
}

register_ideep_operator!{
    GenerateProposals,
    IDEEPFallbackOp::<GenerateProposalsOp::<CPUContext>>
}

register_ideep_operator!{
    GenerateProposalsCPP,
    IDEEPFallbackOp::<GenerateProposalsOp::<CPUContext>>
}

register_ideep_operator!{
    CollectAndDistributeFpnRpnProposals,
    IDEEPFallbackOp::<CollectAndDistributeFpnRpnProposalsOp::<CPUContext>>
}

register_ideep_operator!{
    BoxWithNMSLimit,
    IDEEPFallbackOp::<BoxWithNMSLimitOp::<CPUContext>,SkipIndices::<0,1,2>>
}

register_ideep_operator!{
    BBoxTransform,
    IDEEPFallbackOp::<BBoxTransformOp::<f32,CPUContext>>
}

register_ideep_operator!{
    AffineChannel,
    IDEEPFallbackOp::<AffineChannelOp::<f32,CPUContext>>
}

register_ideep_operator!{
    StopGradient,
    IDEEPFallbackOp::<StopGradientOp::<CPUContext>>
}

register_ideep_operator!{
    PadImage,
    IDEEPFallbackOp::<PadImageOp::<f32,CPUContext>>
}

register_ideep_operator!{
    PRelu,
    IDEEPFallbackOp::<PReluOp::<f32,CPUContext>>
}

// ctc decoder operators
register_ideep_operator!{
    CTCGreedyDecoder,
    IDEEPFallbackOp::<CTCGreedyDecoderOp::<CPUContext>>
}

register_ideep_operator!{
    CTCBeamSearchDecoder,
    IDEEPFallbackOp::<CTCBeamSearchDecoderOp::<CPUContext>>
}

register_ideep_operator!{
    AveragedLossGradient,
    IDEEPFallbackOp::<AveragedLossGradient::<f32,CPUContext>>
}

register_ideep_operator!{
    LabelCrossEntropyGradient,
    IDEEPFallbackOp::<LabelCrossEntropyGradientOp::<f32,CPUContext>>
}

register_ideep_operator!{
    SoftmaxGradient,
    IDEEPFallbackOp::<SoftmaxGradientOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Iter,
    IDEEPFallbackOp::<IterOp::<CPUContext>>
}

register_ideep_operator!{
    LearningRate,
    IDEEPFallbackOp::<LearningRateOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Abs,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,AbsFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Atan,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,AtanFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Sqrt,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,SqrtFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Sign,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,SignFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Div,
    IDEEPFallbackOp::<BinaryElementwiseOp::<NumericTypes,CPUContext,DivFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Mul,
    IDEEPFallbackOp::<BinaryElementwiseOp::<NumericTypes,CPUContext,MulFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Sub,
    IDEEPFallbackOp::<BinaryElementwiseOp::<NumericTypes,CPUContext,SubFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Tanh,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,TanhFunctor::<CPUContext>>>
}

register_ideep_operator!{
    L1Distance,
    IDEEPFallbackOp::<L1DistanceOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Scale,
    IDEEPFallbackOp::<ScaleOp::<CPUContext>>
}

register_ideep_operator!{
    Accuracy,
    IDEEPFallbackOp::<AccuracyOp::<f32,CPUContext>>
}

register_ideep_operator!{
    AddGradient,
    IDEEPFallbackOp::<BinaryElementwiseGradientOp::<NumericTypes,CPUContext,AddFunctor::<CPUContext>>>
}

register_ideep_operator!{
    TanhGradient,
    IDEEPFallbackOp::<BinaryElementwiseOp::<TensorTypes::<f32>,CPUContext,TanhGradientFunctor::<CPUContext>>>
}

register_ideep_operator!{
    MulGradient,
    IDEEPFallbackOp::<BinaryElementwiseGradientOp::<NumericTypes,CPUContext,MulFunctor::<CPUContext>>>
}

register_ideep_operator!{
    TensorProtosDBInput,
    IDEEPFallbackOp::<TensorProtosDBInput::<CPUContext>>
}

register_ideep_operator!{
    CloseBlobsQueue,
    IDEEPFallbackOp::<CloseBlobsQueueOp::<CPUContext>>
}

register_ideep_operator!{
    SoftmaxWithLoss,
    IDEEPFallbackOp::<SoftmaxWithLossOp::<f32,CPUContext>>
}

register_ideep_operator!{
    SoftmaxWithLossGradient,
    IDEEPFallbackOp::<SoftmaxWithLossGradientOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Expand,
    IDEEPFallbackOp::<ExpandOp::<TensorTypes::<i32,i64,f32,double>,CPUContext>>
}

register_ideep_operator!{
    Gather,
    IDEEPFallbackOp::<GatherOp::<CPUContext>>
}

register_ideep_operator!{
    Normalize,
    IDEEPFallbackOp::<NormalizeOp::<f32,CPUContext>>
}

register_ideep_operator!{
    ReduceL2,
    IDEEPFallbackOp::<ReduceOp::<TensorTypes::<f32>,CPUContext,L2Reducer::<CPUContext>>>
}

register_ideep_operator!{
    ReduceSum,
    IDEEPFallbackOp::<ReduceOp::<TensorTypes::<i32,i64,f32,f64>,CPUContext,SumReducer::<CPUContext>>>
}

register_ideep_operator!{
    ReduceMean,
    IDEEPFallbackOp::<ReduceOp::<TensorTypes::<f32>,CPUContext,MeanReducer::<CPUContext>>>
}

register_ideep_operator!{
    BatchMatMul,
    IDEEPFallbackOp::<BatchMatMulOp::<CPUContext>>
}

register_ideep_operator!{
    CreateCommonWorld,
    IDEEPFallbackOp::<CreateCommonWorld::<CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    CloneCommonWorld,
    IDEEPFallbackOp::<CloneCommonWorld::<CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    DestroyCommonWorld,
    IDEEPFallbackOp::<DestroyCommonWorld>
}

register_ideep_operator!{
    Broadcast,
    IDEEPFallbackOp::<BroadcastOp::<CPUContext>>
}

register_ideep_operator!{
    Allreduce,
    IDEEPFallbackOp::<AllreduceOp::<CPUContext>>
}

register_ideep_operator!{
    Allgather,
    IDEEPFallbackOp::<AllgatherOp::<CPUContext>>
}

register_ideep_operator!{
    Barrier,
    IDEEPFallbackOp::<BarrierOp::<CPUContext>>
}

register_ideep_operator!{
    ReduceScatter,
    IDEEPFallbackOp::<ReduceScatterOp::<CPUContext>>
}
