crate::ix!();

#[test] fn sparse_dropout_with_replacement_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SparseDropoutWithReplacement",
        ["X", "Lengths"],
        ["Y", "OutputLengths"],
        ratio=0.5,
        replacement_value=-1
    )

    workspace.FeedBlob("X", np.array([1, 2, 3, 4, 5]).astype(np.int64))
    workspace.FeedBlob("Lengths", np.array([2, 3]).astype(np.int32))
    print("X:", workspace.FetchBlob("X"))
    print("Lengths:", workspace.FetchBlob("Lengths"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("OutputLengths:", workspace.FetchBlob("OutputLengths"))

    X: [1, 2, 3, 4, 5]
    Lengths: [2, 3]
    Y: [1, 2, -1]
    OutputLengths: [2, 1]
    */
}

