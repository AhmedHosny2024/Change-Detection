
import numpy as np
import imageio
import time
import cv2

def stad_img(img, channel_first=True, get_para=False):
    """
    normalization image
    :param channel_first:
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
        center = img - mean  # (channel, height * width)
        var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (height * width, channel)
        mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
        center = img - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (img_height * img_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    if get_para:
        return nm_img, mean, std
    else:
        return nm_img
    
def otsu(data, num=400, get_bcm=False):
    """
    generate binary change map based on otsu
    :param data: cluster data
    :param num: intensity number
    :param get_bcm: bool, get bcm or not
    :return:
        binary change map
        selected threshold
    """
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[1]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    if get_bcm:
        bwp = np.zeros(data.shape)
        bwp[data <= best_threshold] = 0
        bwp[data > best_threshold] = 255
        return bwp, best_threshold
    else:
        return best_threshold

def CVA(img_X, img_Y, stad=False):
    # CVA has not affinity transformation consistency, so it is necessary to normalize multi-temporal images to
    # eliminate the radiometric inconsistency between them
    if stad:
        img_X = stad_img(img_X)
        img_Y = stad_img(img_Y)
    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    return L2_norm


def main():

    # data_set_X = gdal.Open('../../../Dataset/Landsat/Taizhou/2000TM')  # data set X
    # data_set_Y = gdal.Open('../../../Dataset/Landsat/Taizhou/2003TM')  # data set Y

    # img_width = data_set_X.RasterXSize  # image width
    # img_height = data_set_X.RasterYSize  # image height

    # img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    # img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)
    data_set_X = cv2.imread('E:/College/Fourth Year/second term/RSSI/Project/Change-Detection/trainval/A/0069.png')[:, :, 0:3]
    data_set_Y = cv2.imread('E:/College/Fourth Year/second term/RSSI/Project/Change-Detection/trainval/B/0069.png')[:, :, 0:3]
    
    img_X = np.transpose(data_set_X, (2, 0, 1))
    img_Y = np.transpose(data_set_Y, (2, 0, 1))

    channel, img_height, img_width = img_X.shape
    tic = time.time()
    L2_norm = CVA(img_X, img_Y,stad=True)

    bcm = np.zeros((img_height, img_width))
    thre = otsu(L2_norm.reshape(1, -1))
    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    cv2.imwrite('CVA_Taizhou.png', bcm)
    toc = time.time()
    print(toc - tic)


if __name__ == '__main__':
    main()