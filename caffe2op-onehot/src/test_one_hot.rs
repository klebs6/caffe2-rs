crate::ix!();

#[test] fn one_hot_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "OneHot",
        ["indices", "index_size_tensor"],
        ["one_hots"],
    )

    workspace.FeedBlob("indices", np.array([0,1,2,3,4]).astype(np.long))
    print("indices:\n", workspace.FetchBlob("indices"))

    workspace.FeedBlob("index_size_tensor", np.array([5]).astype(np.long))
    print("index_size_tensor:\n", workspace.FetchBlob("index_size_tensor"))

    workspace.RunOperatorOnce(op)
    print("one_hots: \n", workspace.FetchBlob("one_hots"))

    indices:
     [0 1 2 3 4]
    index_size_tensor:
     [5]
    one_hots:
     [[1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1.]]
    */
}
