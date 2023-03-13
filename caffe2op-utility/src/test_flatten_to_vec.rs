crate::ix!();

#[test] fn flatten_to_vec_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "FlattenToVec",
        ["input"],
        ["output"],
    )

    workspace.FeedBlob("input", np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).astype(np.float32))
    print("input:\n", workspace.FetchBlob("input"))

    workspace.RunOperatorOnce(op)
    print("output: \n", workspace.FetchBlob("output"))

    input:
     [[ 1.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]
     [10. 11. 12.]]
    output:
     [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]

    */
}
