crate::ix!();

#[test] fn gather_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Gather",
        ["DATA", "INDICES"],
        ["OUTPUT"]
    )
    data = np.array([[1., 1.2],[2.3, 3.4],[4.5, 5.7]])
    print("DATA:\n",data)

    inds = np.array([[0, 1],[1, 2]])
    print("INDICES:\n",inds)

    // Feed X into workspace
    workspace.FeedBlob("DATA", data.astype(np.float32))
    workspace.FeedBlob("INDICES", inds.astype(np.int32))

    workspace.RunOperatorOnce(op)
    print("OUTPUT:\n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [[1.  1.2]
     [2.3 3.4]
     [4.5 5.7]]
    INDICES:
     [[0 1]
     [1 2]]
    OUTPUT:
     [[[1.  1.2]
      [2.3 3.4]]

     [[2.3 3.4]
      [4.5 5.7]]]

    */
}
