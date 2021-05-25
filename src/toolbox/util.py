# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         util
# Description:  此文件用于实现针对这个比赛的一些必要功能函数
# Author:       Administrator
# Date:         2021/5/20
# -------------------------------------------------------------------------------
import torch.nn.functional as F


#####################################################################
# 验证部分使用相关函数
def np_dice_score_perbatch(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()
    overlap = (p * t).sum()
    return overlap, uion


def loss_fn_train(y_pred, y_true):
    focal = loss_focal(y_pred, y_true)
    dicse = loss_dicse(y_pred, y_true)
    # lovaz = loss_lovaz(y_pred, y_true)

    bcsls = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="mean")
    return 0.8 * focal + 0.6 * bcsls + 0.2 * dicse


def loss_fn_classifier(y_pred, y_true):
    bcsls = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="mean")
    return bcsls


def loss_fn_val(y_pred, y_true):
    val_dice = loss_dicse(y_pred, y_true)
    return val_dice

#######################################################################
# 如下部分为数据集读入服务
class ExternalDataset(D.Dataset):
    def __init__(self, image_dir, mask_dir, transform, threshold=DataSet_ThreShold):
        super(ExternalDataset, self).__init__()
        print("Now Processing External Data")
        self.imgpath = pathlib.Path(image_dir)
        self.mskpath = pathlib.Path(mask_dir)
        self.transform = transform
        self.threshold = threshold
        self.x, self.y, self.z = [], [], []  # x 存放图片 y存放标签 z存放分类器信息
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        for (image_path, mask_path) in zip(self.imgpath.glob('*.png'), self.mskpath.glob('*.png')):
            # 检测文件名是否一致
            image_name = str(image_path).split("/")[-1]
            mask_name = str(mask_path).split("/")[-1]
            # print("image_name:{}".format(image_name))
            # print("mask_name:{}".format(mask_name))
            if (image_name != mask_name):
                print("Error")
                raise ("Order error for external data!!!!!")
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # CHAGE color mode  读入的是  0 到  255 的图像
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 读入的只有  0 和1
            # test_mask = np.unique(mask)
            # if (len(test_mask) != 1):
            #     print(test_mask)
            # print(image.shape)
            # print(mask.shape)
            # Only For test
            if image.shape[-1] != 3:
                raise ("Error")
            if len(mask.shape) != 2:
                raise ("Mask shape error")
            if (mask.shape != image.shape[0:2]):
                raise ("Shape doesnt match!!!")

            ## 加入分类器判断  如果mask中有标记 说明 图中存在肾小球
            if Open_Classifer:
                if mask.sum() >= self.threshold:
                    # 说明该图中有肾小球 对应的分类器为1
                    self.z.append(torch.tensor(1.))
                else:
                    self.z.append(torch.tensor(0.))
            image = cv2.resize(image, (NEW_SIZE, NEW_SIZE))
            mask = np.array(mask, dtype=np.uint8)
            mask = cv2.resize(mask, (NEW_SIZE, NEW_SIZE))
            # ONly for test
            # test_mask = np.unique(mask)
            # if (len(test_mask) != 1):
            #     print(test_mask)
            # print(image.shape)
            # print(mask.shape)
            self.x.append(image)
            self.y.append(mask)

    # get data operation
    def __getitem__(self, index):
        if Open_Classifer:
            image, mask, classifier = self.x[index], self.y[index], self.z[index]
        else:
            image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        if Open_Classifer:
            return self.as_tensor(augments['image']), augments['mask'][None], classifier[None]  # 补入batch维度 方便进行检测
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class HubDataset(D.Dataset):

    def __init__(self, root_dir, tiff_ids, transform,
                 window=256, overlap=32, threshold=DataSet_ThreShold, isvalid=False):
        self.path = pathlib.Path(root_dir)
        self.tiff_ids = tiff_ids  # 输入的是一个列表 表示这次处理的是哪个文件
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'TrainPersudo_0.937_Mix.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold
        self.isvalid = isvalid
        self.x, self.y, self.z = [], [], []
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        print("Now Processing {}".format(self.tiff_ids))
        for i, filename in enumerate(self.csv.index.values):
            if not filename in self.tiff_ids:
                continue
            filepath = (self.path / 'train' / (filename + '.tiff')).as_posix()

            with rasterio.open(filepath, transform=IDNT) as dataset:
                total_mask = rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape)
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)
                if dataset.count != 3:
                    print('Image file with subdatasets as channels:{}'.format(filename))
                    layers = [rasterio.open(subd) for subd in dataset.subdatasets]

                for index, (slc) in enumerate(tqdm(slices)):
                    x1, x2, y1, y2 = slc

                    if dataset.count == 3:  # normal
                        image = dataset.read([1, 2, 3],
                                             window=Window.from_slices((x1, x2), (y1, y2)))
                        image = np.moveaxis(image, 0, -1)
                    else:  # with subdatasets/layers
                        image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
                        for fl in range(3):
                            image[:, :, fl] = layers[fl].read(window=Window.from_slices((x1, x2), (y1, y2)))
                    image = cv2.resize(image, (NEW_SIZE, NEW_SIZE))
                    mask = cv2.resize(total_mask[x1:x2, y1:y2], (NEW_SIZE, NEW_SIZE))
                    # print(image.shape)
                    # print(mask.shape)
                    # print("Processing {} / {} in {} : \n ImageShape:{} MaskShape:{}".format(index + 1, len(slices), filename,image.shape,mask.shape))
                    # 对于测试集，我们应当包括所有可能的边界，因为测试部分不需要进行数据提炼
                    if self.isvalid:
                        self.x.append(image)
                        self.y.append(mask)
                        if Open_Classifer:
                            if total_mask[x1:x2, y1:y2].sum() >= self.threshold:
                                # 说明该图中有肾小球 对应的分类器为1
                                self.z.append(torch.tensor(1.))
                            else:
                                self.z.append(torch.tensor(0.))
                    else:
                        # 阈值判定， 对于训练集，包括边界的图片 或者几乎没有标签的数据 并不是我们需要关注的对象，因此这里我们需要做一个过滤 避免背景数据集过多导致训练失衡
                        # 确保mask中的 标签和
                        # 这里的后一句话 也将某些边界加入了图片中
                        if total_mask[x1:x2, y1:y2].sum() >= self.threshold or (
                                image > 40).mean() > 0.99:  # 4月19日修正 不再限制输入的图片
                            # if total_mask[x1:x2, y1:y2].sum() >= self.threshold:
                            self.x.append(image)
                            self.y.append(mask)
                            if Open_Classifer:
                                if total_mask[x1:x2, y1:y2].sum() >= self.threshold:
                                    # 说明该图中有肾小球 对应的分类器为1
                                    self.z.append(torch.tensor(1.))
                                else:
                                    self.z.append(torch.tensor(0.))

    # get data operation
    def __getitem__(self, index):
        if Open_Classifer:
            image, mask, classifier = self.x[index], self.y[index], self.z[index]
        else:
            image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        if Open_Classifer:
            return self.as_tensor(augments['image']), augments['mask'][None], classifier[None]
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

