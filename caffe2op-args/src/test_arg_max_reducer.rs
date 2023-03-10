crate::ix!();

#[test] fn arg_max_reducer_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ArgMax",
        ["X"],
        ["Indices"],
        axis=2,
        keepdims=False
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(3,3,3))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Indices:", workspace.FetchBlob("Indices"))

    X: [[[4. 9. 6.]
      [6. 6. 1.]
      [9. 5. 4.]]

     [[6. 7. 4.]
      [7. 9. 1.]
      [3. 2. 8.]]

     [[3. 4. 6.]
      [5. 2. 7.]
      [1. 5. 7.]]]
    Indices: [[1 0 0]
     [1 1 2]
     [2 2 2]]

    */
}
