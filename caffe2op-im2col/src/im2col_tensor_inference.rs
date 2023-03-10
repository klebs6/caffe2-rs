crate::ix!();

pub fn im2col_tensor_inference_function(def: &OperatorDef, input: &Vec<TensorShape>) {
    todo!();
    /*
    ArgumentHelper helper(def);
    auto pad = helper.GetSingleArgument<int>("pad", 0);
    auto kernel_h = helper.GetSingleArgument<int>(
        "kernel_h", helper.GetSingleArgument<int>("kernel", 0));
    auto kernel_w = helper.GetSingleArgument<int>(
        "kernel_w", helper.GetSingleArgument<int>("kernel", 0));
    auto dilation_h = helper.GetSingleArgument<int>(
        "dilation_h", helper.GetSingleArgument<int>("dilation", 1));
    auto dilation_w = helper.GetSingleArgument<int>(
        "dilation_w", helper.GetSingleArgument<int>("dilation", 1));
    auto stride_h = helper.GetSingleArgument<int>(
        "stride_h", helper.GetSingleArgument<int>("stride", 1));
    auto stride_w = helper.GetSingleArgument<int>(
        "stride_w", helper.GetSingleArgument<int>("stride", 1));
    auto order = StringToStorageOrder(
        helper.GetSingleArgument<string>("order", "NCHW"));

    const TensorShape& X = in[0];
    int N = 0, C = 0, H = 0, W = 0;
    switch (order) {
        case StorageOrder::NCHW:
            N = X.dims(0);
            C = X.dims(1);
            H = X.dims(2);
            W = X.dims(3);
            break;
        case StorageOrder::NHWC:
            N = X.dims(0);
            H = X.dims(1);
            W = X.dims(2);
            C = X.dims(3);
            break;
        default:
            CAFFE_THROW("Unknown storage order: ", order);
    }

    const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
    const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
    CAFFE_ENFORCE(H >= dkernel_h);
    CAFFE_ENFORCE(W >= dkernel_w);
    const int out_h = (H + 2 * pad - dkernel_h) / stride_h + 1;
    const int out_w = (W + 2 * pad - dkernel_w) / stride_w + 1;

    vector<TensorShape> out(1);
    switch (order) {
        case StorageOrder::NCHW:
            out[0] = CreateTensorShape(
                vector<int>{N, C * kernel_h * kernel_w, out_h, out_w},
                TensorProto::FLOAT);
            break;
        case StorageOrder::NHWC:
            out[0] = CreateTensorShape(
                vector<int>{N, out_h, out_w, kernel_h * kernel_w * C},
                TensorProto::FLOAT);
            break;
        default:
            CAFFE_THROW("Unknown storage order: ", order);
    }

    return out;
    */
}
