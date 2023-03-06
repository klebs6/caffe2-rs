crate::ix!();

#[test] fn empty_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "IsEmpty",
        ["tensor"],
        ["is_empty"],
    )

    // Use a not-empty tensor
    workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("is_empty: ", workspace.FetchBlob("is_empty"),"\n")

    // Use an empty tensor
    workspace.FeedBlob("tensor", np.empty(0))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("is_empty: ", workspace.FetchBlob("is_empty"))

    tensor:
     [[ 0.26018378  0.6778789 ]
     [-1.3097627  -0.40083608]]
    is_empty:  False

    tensor:
     []
    is_empty:  True
    */
}
