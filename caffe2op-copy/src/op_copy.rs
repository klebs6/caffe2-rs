crate::ix!();

use crate::{
    GPUContext,
    GradientMakerBase,
    OperatorStorage,
    CPUContext,
    OperatorDef,
};

#[test] fn copy_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Copy",
        ["input"],
        ["output"]
    )

    workspace.FeedBlob("input", np.random.rand(3,3))
    print("input:", workspace.FetchBlob("input"))
    workspace.RunOperatorOnce(op)
    print("output:", workspace.FetchBlob("output"))


    input:
    [[0.16826761 0.68168217 0.55196001]
     [0.19735483 0.34837823 0.69015595]
     [0.09448514 0.57390828 0.37097193]]
    output:
    [[0.16826761 0.68168217 0.55196001]
     [0.19735483 0.34837823 0.69015595]
     [0.09448514 0.57390828 0.37097193]]

    */
}

/**
  | Copy input tensor into output, potentially
  | across devices.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/copy_op.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/copy_op.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CopyOp<Context, DstContext, SrcContext> {
    storage: OperatorStorage,
    context: Context,
    phantomA: PhantomData<SrcContext>,
    phantomB: PhantomData<DstContext>,
}

num_inputs!{Copy, 1}

num_outputs!{Copy, 1}

inputs!{Copy, 
    0 => ("input", "(*Tensor*): input tensor to copy")
}

outputs!{Copy, 
    0 => ("output", "(*Tensor*): copy of input tensor")
}

identical_type_and_shape!{Copy}

inputs_can_cross_devices!{Copy}

inherit_onnx_schema!{"Identity"}

impl<Context, DstContext, SrcContext> CopyOp<Context, DstContext, SrcContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = this->template Input<Tensor>(0, SrcContext::GetDeviceType());
        auto* output =
            this->template Output<Tensor>(0, DstContext::GetDeviceType());
        output->ResizeLike(input);
        this->context_.template CopyItems<SrcContext, DstContext>(
            input.dtype(),
            input.numel(),
            input.raw_data(),
            output->raw_mutable_data(input.dtype()));
        return true;
        */
    }
}

/**
  | Copy input tensor into output to the
  | specific device
  |
  */
pub struct CopyOnDeviceLikeOp<Context, DstContext, SrcContext> {
    base: CopyOp<Context, DstContext, SrcContext>,
    phantomA: PhantomData<SrcContext>,
    phantomB: PhantomData<DstContext>,
}

num_inputs!{CopyOnDeviceLike, 2}

num_outputs!{CopyOnDeviceLike, 1}

inputs!{CopyOnDeviceLike, 
    0 => ("input", "The input tensor."),
    1 => ("dst", "Tensor, on which device the copy will be performed.")
}

outputs!{CopyOnDeviceLike, 
    0 => ("output", "Tensor that will contain a copy of the input.")
}

impl<Context,DstContext,SrcContext> CopyOnDeviceLikeOp<Context,DstContext,SrcContext> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CopyOp<Context, DstContext, SrcContext>(std::forward<Args>(args)...)
        */
    }
}

/**
  | Take a CPU input tensor and copy it to
  | an output in the current
  | 
  | Context (GPU or CPU). This may involves
  | cross-device MemCpy.
  |
  */
pub type CopyFromCPUInputOp = CopyOp<CPUContext, CPUContext, CPUContext>;

num_inputs!{CopyFromCPUInput, 1}

num_outputs!{CopyFromCPUInput, 1}

inputs!{CopyFromCPUInput, 
    0 => ("input", "The input CPU tensor.")
}

outputs!{CopyFromCPUInput, 
    0 => ("output", "either a TensorCUDA or a TensorCPU")
}

identical_type_and_shape!{CopyFromCPUInput}

inputs_can_cross_devices!{CopyFromCPUInput}

device_inference_function!{CopyFromCPUInput, 
    |def: &OperatorDef| {
        /*
          auto op_device = def.has_device_option() ? def.device_option() : DeviceOption();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), cpu_option);
          vector<DeviceOption> out_dev(def.output_size(), op_device);
          return std::make_pair(in_dev, out_dev);
        */
    }
}

// From CPU, copy it to whatever the current context
register_cpu_operator!{
    CopyFromCPUInput,
    CopyOp<CPUContext, CPUContext, CPUContext>
}

register_cpu_operator!{
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CPUContext, CPUContext, CPUContext>
}

register_cpu_operator!{
    Copy, 
    CopyOp<CPUContext, CPUContext, CPUContext>
}

/**
  | Copy tensor for GPU to CPU context. Must
  | be run under GPU device option.
  |
  */
pub type CopyGPUToCPUOp = CopyOp<CPUContext, GPUContext, CPUContext>;

num_inputs!{CopyGPUToCPU, 1}

num_outputs!{CopyGPUToCPU, 1}

inputs!{CopyGPUToCPU, 
    0 => ("input", "The input tensor.")
}

outputs!{CopyGPUToCPU, 
    0 => ("output", "Tensor that will contain a copy of the input.")
}

identical_type_and_shape!{CopyGPUToCPU}

inputs_can_cross_devices!{CopyGPUToCPU}

device_inference_function!{CopyGPUToCPU, 
    |def: &OperatorDef| {
        todo!();
        /*
          CAFFE_ENFORCE( def.has_device_option(), "CopyGPUToCPU op should have cuda device option.");
          auto& cuda_option = def.device_option();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), cuda_option);
          vector<DeviceOption> out_dev(def.output_size(), cpu_option);
          return std::make_pair(in_dev, out_dev);
        */
    }
}

/**
  | Copy tensor for CPU to GPU context. Must
  | be run under GPU device option.
  |
  */
pub type CopyCPUToGPUOp = CopyOp<CPUContext, CPUContext, GPUContext>;

num_inputs!{CopyCPUToGPU, 1}

num_outputs!{CopyCPUToGPU, 1}

inputs!{CopyCPUToGPU, 
    0 => ("input", "The input tensor.")
}

outputs!{CopyCPUToGPU, 
    0 => ("output", "Tensor that will contain a copy of the input.")
}

identical_type_and_shape!{CopyCPUToGPU}

inputs_can_cross_devices!{CopyCPUToGPU}

device_inference_function!{CopyCPUToGPU, 
    |def: &OperatorDef| {
        todo!();
        /*
          CAFFE_ENFORCE( def.has_device_option(), "CopyCPUToGPU op should have cuda device option.");
          auto& cuda_option = def.device_option();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), cpu_option);
          vector<DeviceOption> out_dev(def.output_size(), cuda_option);
          return std::make_pair(in_dev, out_dev);
        */
    }
}

pub struct GetCopyGradient;

impl GetGradientDefs for GetCopyGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CopyOnDeviceLike",
            "",
            vector<string>{GO(0), I(0)},
            vector<string>{GI(0)});
        */
    }
}

pub struct GetGPUToCPUGradient;

impl GetGradientDefs for GetGPUToCPUGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (g_output_[0].IsDense()) {
          return SingleGradientDef(
              "CopyCPUToGPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
                                         "CopyCPUToGPU",
                                         "",
                                         std::vector<string>{GO_I(0)},
                                         std::vector<string>{GI_I(0)}),
                                     CreateOperatorDef(
                                         "CopyCPUToGPU",
                                         "",
                                         std::vector<string>{GO_V(0)},
                                         std::vector<string>{GI_V(0)})};
        }
        */
    }
}

pub struct GetCPUToGPUGradient;

impl GetGradientDefs for GetCPUToGPUGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (g_output_[0].IsDense()) {
          return SingleGradientDef(
              "CopyGPUToCPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
                                         "CopyGPUToCPU",
                                         "",
                                         std::vector<string>{GO_I(0)},
                                         std::vector<string>{GI_I(0)}),
                                     CreateOperatorDef(
                                         "CopyGPUToCPU",
                                         "",
                                         std::vector<string>{GO_V(0)},
                                         std::vector<string>{GI_V(0)})};
        }
        */
    }
}

register_gradient!{Copy,         GetCopyGradient}
register_gradient!{CopyGPUToCPU, GetGPUToCPUGradient}
register_gradient!{CopyCPUToGPU, GetCPUToGPUGradient}
