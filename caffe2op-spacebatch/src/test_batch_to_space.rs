crate::ix!();

#[test] fn batch_to_space_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BatchToSpace",
        ["X"],
        ["Y"],
        pad=3
    )

    workspace.FeedBlob("X", np.random.rand(10,3,32,32).astype(np.float32))
    print("X.shape:", workspace.FetchBlob("X").shape)
    workspace.RunOperatorOnce(op)
    print("Y.shape:", workspace.FetchBlob("Y").shape)

    X.shape: (10, 3, 32, 32)
    Y.shape: (2, 3, 58, 58)
    */
}
