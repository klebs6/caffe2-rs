crate::ix!();

#[test] fn has_elements_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "HasElements",
        ["tensor"],
        ["has_elements"],
    )

    // Use a not-empty tensor
    workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("has_elements: ", workspace.FetchBlob("has_elements"),"\n")

    // Use an empty tensor
    workspace.FeedBlob("tensor", np.empty(0))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("has_elements: ", workspace.FetchBlob("has_elements"))

    tensor:
     [[ 0.6116506  -0.54433197]
     [ 0.19406661 -0.7338629 ]]
    has_elements:  True

    tensor:
     []
    has_elements:  False
    */
}

