crate::ix!();

#[test] fn mat_mul_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "MatMul",
        ["A", "B"],
        ["Y"],
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.float32))
    workspace.FeedBlob("B", np.random.randint(10, size=(3,3)).astype(np.float32))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    A: [[1. 8. 3.]
     [6. 4. 4.]
     [5. 4. 7.]]
    B: [[4. 0. 3.]
     [3. 1. 1.]
     [8. 5. 8.]]
    Y: [[52. 23. 35.]
     [68. 24. 54.]
     [88. 39. 75.]]

    */
}
