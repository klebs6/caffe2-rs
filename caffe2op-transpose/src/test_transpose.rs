crate::ix!();

#[test] fn transpose_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Transpose",
        ["X"],
        ["Y"],
        axes=(0,3,1,2)
    )

    x = np.random.rand(1,32,32,3)
    workspace.FeedBlob("X", x)
    print("X.shape (NHWC order):", workspace.FetchBlob("X").shape)
    workspace.RunOperatorOnce(op)
    print("Y.shape (NCHW order):", workspace.FetchBlob("Y").shape)

    X.shape (NHWC order): (1, 32, 32, 3)
    Y.shape (NCHW order): (1, 3, 32, 32)
    */
}
