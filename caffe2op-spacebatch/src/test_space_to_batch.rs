crate::ix!();

#[test] fn space_to_batch_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SpaceToBatch",
        ["X"],
        ["Y"],
        pad=2,
        block_size=3
    )

    workspace.FeedBlob("X", np.random.rand(1,3,5,5).astype(np.float32))
    print("X.shape:", workspace.FetchBlob("X").shape)
    workspace.RunOperatorOnce(op)
    print("Y.shape:", workspace.FetchBlob("Y").shape)


    X.shape: (1, 3, 5, 5)
    Y.shape: (9, 3, 3, 3)

    */
}
