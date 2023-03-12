crate::ix!();

#[test] fn append_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Append",
        ["A", "B"],
        ["A"],
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(1,3,3)))
    workspace.FeedBlob("B", np.random.randint(10, size=(2,3,3)))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("A:", workspace.FetchBlob("A"))

    A:
    [[[3 8 7]
      [1 6 6]
      [5 0 6]]]
    B:
    [[[4 3 1]
      [7 9 6]
      [9 4 5]]

     [[7 7 4]
      [9 8 7]
      [1 6 6]]]
    A:
    [[[3 8 7]
      [1 6 6]
      [5 0 6]]

     [[4 3 1]
      [7 9 6]
      [9 4 5]]

     [[7 7 4]
      [9 8 7]
      [1 6 6]]]

    */
}
