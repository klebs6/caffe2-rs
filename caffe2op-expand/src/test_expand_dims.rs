crate::ix!();

#[test] fn expand_dims_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

        op = core.CreateOperator(
            "ExpandDims",
            ["data"],
            ["expanded"],
            dims=[0,1],
        )

        workspace.FeedBlob("data", np.zeros((100,100)).astype(np.float32))
        print("data.shape:", workspace.FetchBlob("data").shape)

        workspace.RunOperatorOnce(op)
        print("expanded.shape:", workspace.FetchBlob("expanded").shape)


        data.shape: (100, 100)
        expanded.shape: (1, 1, 100, 100)
    */
}
