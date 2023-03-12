crate::ix!();

#[test] fn squeeze_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Squeeze",
        ["data"],
        ["squeezed"],
        dims=[0,1],
    )

    workspace.FeedBlob("data", np.zeros((1,1,100,100)).astype(np.float32))
    print("data.shape:", workspace.FetchBlob("data").shape)

    workspace.RunOperatorOnce(op)
    print("squeezed.shape:", workspace.FetchBlob("squeezed").shape)

    data.shape: (1, 1, 100, 100)
    squeezed.shape: (100, 100)
    */
}
