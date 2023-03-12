crate::ix!();

pub fn conv_doc_generator(dim: *const u8) -> doc_fn {

    todo!();
    /*
  return [=](OpSchema& schema) {
    string doc = R"DOC(
The convolution operator consumes an input vector, a {dim}filter blob
and a bias blob and computes the output. {conv_doc})DOC";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{conv_doc}", kConvDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob."
      );
    schema.Input(
        1,
        "filter",
        "The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data."
      );
    schema.Input(
        2,
        "bias",
        "The bias blob, of length $M$, containing the biases for the convolution, one bias per filter."
      );
    schema.Output(
        0,
        "Y",
        "Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution."
      );
      /*
    schema.Arg(
        "kernel",
        "*(type: int; default: 0)* Desired kernel size. If left at default the kernel size will be inferred from the input $filter$ blob.",
        0
    );
    schema.Arg(
        "stride",
        "*(type: int; default: 1)* Controls the stride of the kernel as it traverses the input blob.",
        0
    );
    schema.Arg(
        "dilation",
        "*(type: int; default: 1)* Controls spacing between kernel points. If dilation is greater than one, the kernel does not operate on a contiguous spatial region. For a visualization click [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md).",
        0
    );
    schema.Arg(
        "pad",
        "*(type: int; default: 0)* Controls the amount of padding to apply to the input feature map before computing the convolution.",
        0
    );
    schema.Arg(
        "float16_compute",
        "*(type: bool; default: False)* Whether to use float-16 compute kernel.",
        0
    );
    schema.Arg(
        "group",
        "*(type: int; default: 1)* Controls level of group convolution. For more info click [here](https://blog.yani.io/filter-group-tutorial/).",
        0
    );
    schema.Arg(
        "order",
        "*(type: string; default: \"NCHW\")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is \"NHWC\".",
        0
    );
    schema.Arg(
        "shared_buffer",
        "*(type: int; default: 0)*",
        0
    );
    */
  };

    */
}
