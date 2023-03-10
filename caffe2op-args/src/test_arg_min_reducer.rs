crate::ix!();

#[test] fn arg_min_reducer_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ArgMin",
        ["X"],
        ["Indices"],
        axis=1
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(5,5))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Indices:", workspace.FetchBlob("Indices"))

    X: [[9. 4. 6. 4. 1.]
      [5. 9. 8. 3. 4.]
      [6. 1. 0. 2. 9.]
      [7. 8. 2. 4. 9.]
      [3. 9. 4. 9. 4.]]
    Indices: [[4]
      [3]
      [2]
      [2]
      [0]]
    */
}
