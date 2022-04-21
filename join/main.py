# -*- coding: utf-8 -*- 
# @File main
# @Time 2022/4/20
# @Author wcy
# @Software: PyCharm
# @Site
import time

import cv2
import numpy as np


def resize(image: np.ndarray, tar_w=800):
    h, w, _ = image.shape
    tar_h = int((h * tar_w) / w)
    image = cv2.resize(image, (tar_w, tar_h))
    return image


def sift():
    imgname1 = "imgs/拍照点1-20220412-170415.jpg"
    imgname2 = "imgs/拍照点3-20220412-170517.jpg"

    sift = cv2.xfeatures2d.SIFT_create()

    # FLANN 参数设计
    FLANN_INDEX_KDTREE = 0
    # KTreeIndex配置索引，指定待处理核密度树的数量（理想的数量在1-16）
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # 指定递归遍历的次数
    search_params = dict(checks=50)
    # 快速最近邻搜索包,一个对大数据集和高维特征进行最近邻搜索的算法的集合
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    img1 = cv2.imdecode(np.fromfile(imgname1, dtype=np.uint8), 1)
    img1 = resize(img1, tar_w=1000)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    kp1, des1 = sift.detectAndCompute(gray1, None)  # des是描述子

    img2 = cv2.imdecode(np.fromfile(imgname2, dtype=np.uint8), 1)
    img2 = resize(img2, tar_w=1000)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))

    hmerge = np.hstack((img3, img4))
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])

    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imshow("FLANN", resize(img5))
    # cv2.imshow("point", resize(hmerge))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_homo(img1, img2, dtype="sift", ransacReprojThreshold=5.0):
    if dtype == "sift":
        # 创建特征转换对象
        sift = cv2.xfeatures2d.SIFT_create()
        start = time.time()
        # 获取特征点和描述子
        k1, d1 = sift.detectAndCompute(img1, None)
        k2, d2 = sift.detectAndCompute(img2, None)
        # print("sift", time.time() - start)
        """
        k1 k2是关键点。它所包含的信息有：
        angle：角度，表示关键点的方向，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
        class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
        octave：代表是从金字塔哪一层提取的得到的数据。
        pt：关键点点的坐标
        response：响应程度，代表该点强壮大小，更确切的说，是该点角点的程度。
        size：该点直径的大小
        """
    elif dtype == "surf":
        # pip install opencv-contrib-python==3.4.2.17
        surf = cv2.xfeatures2d.SURF_create(400)
        start = time.time()
        # 找到关键点和描述符
        k1, d1 = surf.detectAndCompute(img1, None)
        k2, d2 = surf.detectAndCompute(img2, None)
        # print("surf", time.time() - start)
    # 创建特征匹配器
    bf = cv2.BFMatcher()
    # 使用描述子进行一对多的描述子匹配
    maches = bf.knnMatch(d1, d2, k=2)
    """
    d1 shape n*128
    d2 shape m*128
    knnMatch 就是拿d1的 所有向量依次跟d2的向量计算距离，并排序，参数k即保留最接近的前k个d2的向量信息
    maches 参数解释
    distance : 为两个描述子之间的距离，越小表示匹配度越高 计算公式: 0ρ = √( (x1-x2)2+(y1-y2)2 )　|x| = √( x2 + y2 )
    queryIdx : 查询点的索引（当前要寻找匹配结果的点在它所在图片上的索引）.类似于序号
    trainIdx : 被查询到的点的索引（存储库中的点的在存储库上的索引）
    imgIdx : 有争议(常为0)
    """
    # 筛选有效的特征描述子存入数组中
    verify_matches = []
    for m1, m2 in maches:
        if m1.distance / m2.distance < 0.8:
            verify_matches.append([m1])
    # 单应性矩阵需要最低四个特征描述子坐标点进行计算，判断数组中是否有足够,这里设为6更加充足
    if len(verify_matches) > 6:
        # 存放求取单应性矩阵所需的img1和img2的特征描述子坐标点
        img1_pts = []
        img2_pts = []
        for m in verify_matches:
            # 通过使用循环获取img1和img2图像优化后描述子所对应的特征点
            img1_pts.append(k1[m[0].queryIdx].pt)
            img2_pts.append(k2[m[0].trainIdx].pt)
        # 得到的坐标是[(x1,y1),(x2,y2),....]
        # 计算需要的坐标格式：[[x1,y1],[x2,y2],....]所以需要转化
        img1_pts = np.array(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.array(img2_pts).reshape(-1, 1, 2)
        # 计算单应性矩阵用来优化特征点
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, ransacReprojThreshold=ransacReprojThreshold)
        drawpara = dict(singlePointColor=None, matchColor=None, flags=2)
        img = cv2.drawMatchesKnn(img1, k1, img2, k2, verify_matches, None, **drawpara)

        # 仅保留平移和旋转
        # H[2:, :] = np.array([[0, 0, 1]])
        return H, img
    else:
        return None, None


def stitch_image(img1, img2, H):
    # 1、获得每张图片的四个角点
    # 2、对图片进行变换，（单应性矩阵使）
    # 3、创建大图，将两张图拼接到一起
    # 4、将结果输出
    # 获取原始图的高、宽
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # 获取四个点的坐标，变换数据类型便于计算
    img1_dims = np.float32([[0, 0], [0, h1], [h1, w1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims = np.float32([[0, 0], [0, h2], [h2, w2], [w2, 0]]).reshape(-1, 1, 2)
    # 获取根据单应性矩阵透视变换后的图像四点坐标
    img1_transform = cv2.perspectiveTransform(img1_dims, H)
    # img2_transform = cv2.perspectiveTransform(img2_dims,H)

    # 合并矩阵  获取最大x和最小x，最大y和最小y  根据最大最小计算合并后图像的大小；
    # #计算方式： 最大-最小
    result_dims = np.concatenate((img2_dims, img1_transform), axis=0)

    # 使用min获取横向最小坐标值，ravel将二维转换为一维，强转为int型，
    # 最小-0.5，最大+0.5防止四舍五入造成影响
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
    # 平移距离
    transform_dist = [-x_min, -y_min]
    # 齐次变换矩阵
    transform_arary = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])
    # 输出图像的尺寸
    ww = x_max - x_min
    hh = y_max - y_min
    # 透视变换实现平移
    result_img = cv2.warpPerspective(img1, transform_arary.dot(H), (ww, hh))
    # 将img2添加进平移后的图像
    result_img[transform_dist[1]:transform_dist[1] + h2,
    transform_dist[0]:transform_dist[0] + w2] = img2
    return result_img


def rotate(image, angle):
    """
    旋转图片
    :param image:
    :param angle:
    :return:
    """
    PI = np.pi
    heightNew = int(
        image.shape[1] * np.abs(np.sin(angle * PI / 180)) + image.shape[0] * np.abs(np.cos(angle * PI / 180)))
    widthNew = int(
        image.shape[0] * np.abs(np.sin(angle * PI / 180)) + image.shape[1] * np.abs(np.cos(angle * PI / 180)))
    pt = (image.shape[1] / 2., image.shape[0] / 2.)
    # 旋转矩阵
    # --                      --
    # | ∂, β, (1 -∂) * pt.x - β * pt.y |
    # | -β, ∂, β * pt.x + (1 -∂) * pt.y |
    # --                      --
    # 其中 ∂=scale * cos(angle), β = scale * sin(angle)

    ### getRotationMatrix2D 的实现 ###
    scale = 1
    a = scale * np.cos(angle * PI / 180)
    b = scale * np.sin(angle * PI / 180)
    r1 = np.array([[a, b, (1 - a) * pt[0] - b * pt[1]],
                   [-b, a, b * pt[0] + (1 - a) * pt[1]]])
    ### getRotationMatrix2D 的实现 ###

    r = cv2.getRotationMatrix2D(pt, angle, 1.0)  # 获取旋转矩阵(旋转中心(pt), 旋转角度(angle)， 缩放系数(scale)
    r[0, 2] += (widthNew - image.shape[1]) / 2
    r[1, 2] += (heightNew - image.shape[0]) / 2
    # 进行仿射变换（输入图像, 2X3的变换矩阵, 指定图像输出尺寸, 插值算法标识符, 边界填充BORDER_REPLICATE)
    dst = cv2.warpAffine(image, r, (widthNew, heightNew), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return dst


indexH = 0
indexW = 0


def onChangeH(x):
    global indexH
    indexH = x


def onChangeW(x):
    global indexW
    indexW = x


def draw_grid(img, line_num=10, color=(0, 255, 0)):
    h, w, _ = img.shape
    unit_h = h // line_num
    unit_w = w // line_num
    for i in range(1, line_num):
        cv2.line(img, (0, i * unit_h), (w, i * unit_h), color, thickness=2)
    for i in range(1, line_num):
        cv2.line(img, (i * unit_w, 0), (i * unit_w, h), color, thickness=2)


def main():
    # imgname1 = "imgs/拍照点1-20220412-170415.jpg"
    # imgname2 = "imgs/拍照点3-20220412-170517.jpg"
    # imgname3 = "imgs/拍照点5-20220412-170623.jpg"

    imgname1 = "imgs/拍照点2-20220412-170453.jpg"
    imgname2 = "imgs/拍照点4-20220412-170548.jpg"
    imgname3 = "imgs/拍照点6-20220412-170720.jpg"
    # 读取两张图片
    img1 = cv2.imdecode(np.fromfile(imgname1, dtype=np.uint8), 1)
    img2 = cv2.imdecode(np.fromfile(imgname2, dtype=np.uint8), 1)
    img3 = cv2.imdecode(np.fromfile(imgname3, dtype=np.uint8), 1)

    img1 = resize(img1, tar_w=800)
    img2 = resize(img2, tar_w=800)
    img3 = resize(img3, tar_w=800)

    img1 = rotate(img1, 0)
    img2 = rotate(img2, 0)
    img3 = rotate(img3, 0)

    win_name = "concat"

    cv2.namedWindow(win_name)
    cv2.createTrackbar("height", win_name, 0, 200, onChangeH)
    cv2.createTrackbar("width", win_name, 0, 200, onChangeW)

    i = -1
    while True:
        i += 1
        img1_grid = np.copy(img1)
        img2_grid = np.copy(img2)
        img3_grid = np.copy(img3)
        draw_grid(img1_grid, color=(0, 225, 0))
        draw_grid(img2_grid, color=(225, 0, 0))
        draw_grid(img3_grid, color=(0, 0, 225))

        ransacReprojThreshold = i % 100 * 0.1
        # start = time.time()
        H, match_img1_2 = get_homo(img1, img2, ransacReprojThreshold=ransacReprojThreshold)
        print(H.tolist()[0])
        print(H.tolist()[1])
        print(H.tolist()[2])
        print()
        img1_2_ = stitch_image(img1, img2, H)
        # print(time.time() - start)
        img1_2 = stitch_image(img1_grid, img2_grid, H)

        H2, match_img2_3 = get_homo(img1_2_, img3)
        result_img = stitch_image(img1_2, img3_grid, H2)

        # img1_zero = np.zeros_like(img1)
        # img2_zero = np.zeros_like(img2)
        # h, w, _ = img2.shape
        # img1_zero[:, :w - indexW, ...] = img1[:, indexW:, ...]
        # img2_zero[:h - indexH, ...] = img2[indexH:, ...]
        # concat = np.vstack((img1_zero, img2_zero))
        # cv2.imshow(win_name, resize(concat, tar_w=600))
        # cv2.imshow("img1", resize(img1, tar_w=500))
        # cv2.imshow("img2", resize(img2, tar_w=500))
        # cv2.imshow("img3", resize(img3, tar_w=500))
        # cv2.imshow("match_img1_2", resize(match_img1_2, tar_w=1000))
        # cv2.imshow("match_img2_3", resize(match_img2_3, tar_w=1000))
        result = resize(result_img, tar_w=600)
        cv2.putText(result, f"{round(ransacReprojThreshold, 3)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow("result", result)
        cv2.waitKey(100)


if __name__ == '__main__':
    # sift()
    main()
