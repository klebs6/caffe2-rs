crate::ix!();

#[test] fn conv_transpose_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ConvTranspose",
        ["X", "filter", "bias"],
        ["Y"],
        kernels=[2,2],
        pads=[4,4,4,4],
        strides=[2,2]
    )

    // Create X: (N,C,H,W)
    data = np.random.randn(2,3,5,5).astype(np.float32)
    print("Data shape: ",data.shape)

    // Create filter: (M,C,Kh,Kw)
    filters = np.random.randn(3,1,2,2).astype(np.float32)
    print("Filter shape: ",filters.shape)

    // Create b: M
    bias = np.array([1.]).astype(np.float32)
    print("Bias shape: ",bias.shape)

    // Put the inputs into the workspace
    workspace.FeedBlob("X", data)
    workspace.FeedBlob("filter", filters)
    workspace.FeedBlob("bias", bias)

    // Run the operator
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))


    result: 

    Data shape:  (2, 3, 5, 5)
    Filter shape:  (3, 1, 2, 2)
    Bias shape:  (1,)
    Y:
     [[[[0.53606427 0.5775447 ]
       [0.40148795 1.5188271 ]]]


     [[[1.9903406  3.2794335 ]
       [0.09960175 0.31917763]]]]

    */
}

