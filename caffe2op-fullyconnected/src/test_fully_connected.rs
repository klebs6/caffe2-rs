crate::ix!();

#[test] fn fully_connected_op_example() {

    todo!();

    /*
    // In this example, our batch size is 1 (M=1), the input observation will have
    //   6 features (K=6), and the layer will have one hidden node (N=1). The
    //   expected output is Y=7.
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "FC",
        ["X", "W", "b"],
        ["Y"]
    )

    // Create X: MxK
    data = np.array([1,2,3,4,5,6]).astype(np.float32)
    data = data[np.newaxis,:]

    // Create W: NxK
    weights = np.array(np.array([1,1/2.,1/3.,1/4.,1/5.,1/6.])).astype(np.float32)
    weights = weights[np.newaxis,:]

    // Create b: N
    bias = np.array([1.]).astype(np.float32)

    // Put the inputs into the workspace
    workspace.FeedBlob("X", data)
    workspace.FeedBlob("W", weights)
    workspace.FeedBlob("b", bias)

    // Run the operator
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    Y:
     [[7.]]

    */
}
