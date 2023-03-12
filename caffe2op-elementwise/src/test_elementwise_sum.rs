crate::ix!();

#[test] fn elementwise_sum_example1() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sum",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([[1,2],[3,4]]).astype(np.float32))
    workspace.FeedBlob("B", np.array([[5,6],[7,8]]).astype(np.float32))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("A"))

    **Result**

    A: [[1. 2.]
     [3. 4.]]
    B: [[5. 6.]
     [7. 8.]]
    C: [[1. 2.]
     [3. 4.]]

    */
}

#[test] fn elementwise_sum_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sum",
        ["A",  "B"],
        ["A"],  // inplace
    )

    workspace.FeedBlob("A", np.array([[1,2,5],[8,3,4]]).astype(np.float32))
    workspace.FeedBlob("B", np.array([[9,5,6],[6,7,8]]).astype(np.float32))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("A after Sum:", workspace.FetchBlob("A"))

    A: [[1. 2. 5.]
     [8. 3. 4.]]
    B: [[9. 5. 6.]
     [6. 7. 8.]]
    A after Sum: [[10.  7. 11.]
     [14. 10. 12.]]

    */
}
