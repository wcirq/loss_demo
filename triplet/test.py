import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def test_pairwise_distances(squared = False):
    """两两embedding的距离，比如第一行， 0和0距离为0， 0和1距离为8， 0和2距离为16 （注意开过根号）
    [[ 0.  8. 16.]
     [ 8.  0.  8.]
     [16.  8.  0.]]
    """
    embeddings = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]], dtype=np.float32)
    dot_product = np.dot(embeddings, np.transpose(embeddings))
    square_norm = np.diag(dot_product)
    distances = np.expand_dims(square_norm, axis=1) - 2.0*dot_product + np.expand_dims(square_norm, 0)
    if not squared:
        mask = np.float32(np.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = np.sqrt(distances)
        distances = distances * (1.0 - mask)
    print(distances)
    return distances


def test_get_triplet_mask(labels):
    '''
    valid （i, j, k）满足
         - i, j, k 不相等
         - labels[i] == labels[j]  && labels[i] != labels[k]

    '''
    # 初始化一个二维矩阵，坐标(i, j)不相等置为1，得到indices_not_equal
    indices_equal = np.cast[np.bool](np.eye(np.shape(labels)[0], dtype=np.int32))
    indices_not_equal = np.logical_not(indices_equal)
    # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
    i_not_equal_j = np.expand_dims(indices_not_equal, 2)
    i_not_equal_k = np.expand_dims(indices_not_equal, 1)
    j_not_equal_k = np.expand_dims(indices_not_equal, 0)
    # 想得到i!=j!=k, 三个不等取and即可
    # 比如这里得到
    '''array([[[False, False, False],
               [False, False,  True],
               [False,  True, False]],
              [[False, False,  True],
               [False, False, False],
               [ True, False, False]],
              [[False,  True, False],
              [ True, False, False],
              [False, False, False]]])'''
    # 只有下标(i, j, k)不相等时才是True
    distinct_indices = np.logical_and(np.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # 同样根据labels得到对应i=j, i!=k
    label_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
    i_equal_j = np.expand_dims(label_equal, 2)
    i_equal_k = np.expand_dims(label_equal, 1)
    valid_labels = np.logical_and(i_equal_j, np.logical_not(i_equal_k))

    # mask即为满足上面两个约束，所以两个3D取and
    mask = np.logical_and(valid_labels, distinct_indices)
    return mask


if __name__ == '__main__':
    test_pairwise_distances()
    A = np.array([[4, 2, -5],
         [6, 4, -9],
         [5, 3, -7]])

    min = A.min()
    max = A.max()
    A = (A-min)/(max-min)

    pca = PCA(n_components=2)
    pca.fit(A.T)

    c_w, c_h = 400, 400
    canvas = np.zeros((c_h, c_w))

    p = np.array([[1, 1, 1], [1, 3, -2]])
    # p = np.array([[1, 3, 2], [1, 3, 2]])
    B = np.dot(p, A).T
    B2 = pca.transform(A.T)

    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.scatter3D(A[0, :], A[1, :], A[2, :], cmap='Blues')
    ax1.scatter3D(B[:, 0], B[:, 1], 0, cmap='Red')
    ax1.scatter3D(B2[:, 0], B2[:, 1], 0, cmap='GREEN')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()
