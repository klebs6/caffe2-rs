crate::ix!();

#[test] fn top_k_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "TopK",
        ["X"],
        ["Values", "Indices", "Flattened_indices"],
        k=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(3,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Values:", workspace.FetchBlob("Values"))
    print("Indices:", workspace.FetchBlob("Indices"))
    print("Flattened_indices:", workspace.FetchBlob("Flattened_indices"))

    X:
    [[[6. 7. 0.]
      [8. 7. 7.]
      [1. 5. 6.]]

     [[0. 6. 1.]
      [2. 8. 4.]
      [1. 2. 9.]]

     [[4. 3. 7.]
      [0. 1. 7.]
      [0. 1. 8.]]]
    Values:
    [[[7. 6.]
      [8. 7.]
      [6. 5.]]

     [[6. 1.]
      [8. 4.]
      [9. 2.]]

     [[7. 4.]
      [7. 1.]
      [8. 1.]]]
    Indices:
    [[[1 0]
      [0 1]
      [2 1]]

     [[1 2]
      [1 2]
      [2 1]]

     [[2 0]
      [2 1]
      [2 1]]]
    Flattened_indices: [ 1  0  3  4  8  7 10 11 13 14 17 16 20 18 23 22 26 25]
    */
}

