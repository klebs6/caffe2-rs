crate::ix!();

#[test] fn sum_elements_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    sum_op = core.CreateOperator(
        "SumElements",
        ["X"],
        ["Y"]
    )

    avg_op = core.CreateOperator(
        "SumElements",
        ["X"],
        ["Y"],
        average=True
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(sum_op)
    print("Y (sum_op):", workspace.FetchBlob("Y"))
    workspace.RunOperatorOnce(avg_op)
    print("Y (avg_op):", workspace.FetchBlob("Y"))

    X:
     [[7. 2. 5.]
     [9. 4. 2.]
     [1. 2. 5.]]
    Y (sum_op): 37.0
    Y (avg_op): 4.111111
    */
}
