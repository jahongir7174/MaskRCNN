import math
import random
from copy import deepcopy
from os.path import basename

import cv2
import mmcv
import numpy
from PIL import Image, ImageOps, ImageEnhance
from mmdet.datasets.pipelines.transforms import PIPELINES

max_value = 10.

# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)


def resample():
    return random.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC))


def resize(image, image_size):
    h, w = image.shape[:2]
    ratio = image_size / max(h, w)
    if ratio != 1:
        shape = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, shape, interpolation=resample())
    return image, image.shape[:2]


def xy2wh(x):
    y = numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyn2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * x[:, 0] + pad_w  # top left x
    y[:, 1] = h * x[:, 1] + pad_h  # top left y
    return y


def whn2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def mask2box(mask, w, h):
    x, y = mask.T

    inside = (x >= 0) & (y >= 0) & (x < w) & (y < h)

    x = x[inside]
    y = y[inside]

    if any(x) and any(y):
        return numpy.array([x.min(), y.min(), x.max(), y.max()]), x, y
    else:
        return numpy.zeros((1, 4)), x, y


def box_ioa(box1, box2):
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    area1 = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0)
    area2 = (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0)

    # Intersection over area
    return (area1 * area2) / ((b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-7)


def masks2boxes(masks):
    boxes = []
    for mask in masks:
        x, y = mask.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xy2wh(numpy.array(boxes))


def resample_masks(masks, n=1000):
    for i, s in enumerate(masks):
        s = numpy.concatenate((s, s[0:1, :]), axis=0)
        x = numpy.linspace(0, len(s) - 1, n)
        xp = numpy.arange(len(s))
        mask = [numpy.interp(x, xp, s[:, i]) for i in range(2)]
        masks[i] = numpy.concatenate(mask).reshape(2, -1).T
    return masks


def box_candidates(box1, box2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    area = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.01) & (area < 20)


def copy_paste(image, boxes, masks, p=0.):
    # Copy-Paste augmentation https://arxiv.org/abs/2012.07177
    n = len(masks)
    if p and n:
        h, w, c = image.shape
        img = numpy.zeros(image.shape, numpy.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = boxes[j], masks[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = box_ioa(box, boxes[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                boxes = numpy.concatenate((boxes, [[l[0], *box]]), 0)
                masks.append(numpy.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(img, [masks[j].astype(numpy.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=image, src2=img)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        image[i] = result[i]

    return image, boxes, masks


def random_hsv(image):
    # HSV color-space augmentation
    r = numpy.random.uniform(-1, 1, 3) * [0.015, 0.7, 0.4] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * r[2], 0, 255).astype('uint8')

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def random_perspective(image, boxes, masks, size):
    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-0, 0)
    s = random.uniform(1 - 0.5, 1 + 0.5)
    rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-0, 0) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-0, 0) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * size  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * size  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    if (matrix != numpy.eye(3)).any():
        image = cv2.warpAffine(image, matrix[:2], dsize=(size, size))  # affine

    if len(boxes):
        new_masks = []
        new_boxes = numpy.zeros((len(boxes), 4))
        for i, mask in enumerate(resample_masks(masks)):
            xy = numpy.ones((len(mask), 3))
            xy[:, :2] = mask
            xy = xy @ matrix.T
            xy = xy[:, :2]

            # clip
            new_boxes[i], x, y = mask2box(xy, size, size)
            new_masks.append([x, y])

        # filter candidates
        candidates = box_candidates(boxes[:, 1:5].T * s, new_boxes.T)
        boxes = boxes[candidates]
        boxes[:, 1:5] = new_boxes[candidates]
        masks = []
        for candidate, new_mask in zip(candidates, new_masks):
            if candidate:
                masks.append(new_mask)
    return image, boxes, masks


def mosaic(self, index, size=None):
    if size is None:
        size = numpy.random.choice(self.image_sizes)

    xc = int(random.uniform(size // 2, 2 * size - size // 2))
    yc = int(random.uniform(size // 2, 2 * size - size // 2))

    indexes4 = [index] + random.choices(range(self.num_samples), k=3)
    numpy.random.shuffle(indexes4)

    results4 = [deepcopy(self.dataset[index]) for index in indexes4]
    filename = results4[0]['filename']

    boxes4 = []
    masks4 = []
    shapes = [x['img_shape'][:2] for x in results4]
    image4 = numpy.full((2 * size, 2 * size, 3), 0, numpy.uint8)

    for i, (results, shape) in enumerate(zip(results4, shapes)):
        image, (h, w) = resize(results['img'], size)

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size * 2), min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        masks = []
        label = numpy.array(results['ann_info']['labels'])

        for mask in results['ann_info']['masks']:
            mask = [j for i in mask for j in i]
            mask = numpy.array(mask).reshape(-1, 2)
            masks.append(mask / numpy.array([shape[1], shape[0]]))

        boxes = (label.reshape(-1, 1), masks2boxes(masks))
        boxes = numpy.concatenate(boxes, axis=1)

        if len(boxes):
            boxes[:, 1:] = whn2xy(boxes[:, 1:], w, h, pad_w, pad_h)
            masks = [xyn2xy(x, w, h, pad_w, pad_h) for x in masks]

        boxes4.append(boxes)
        masks4.extend(masks)

    # concatenate & clip
    boxes4 = numpy.concatenate(boxes4, 0)

    for box4 in boxes4[:, 1:]:
        numpy.clip(a=box4, a_min=0, a_max=2 * size, out=box4)

    for mask4 in masks4:
        numpy.clip(a=mask4, a_min=0, a_max=2 * size, out=mask4)

    image4, boxes4, masks4 = copy_paste(image4, boxes4, masks4, p=0.25)
    image4, boxes4, masks4 = random_perspective(image4, boxes4, masks4, size)

    label = []
    boxes = []
    masks = []
    for box4, mask4 in zip(boxes4, masks4):
        if len(mask4[0]) != len(mask4[1]):
            return None
        if len(mask4[0]) % 2 != 0 or len(mask4[0]) < 6:
            return None
        if len(mask4[1]) % 2 != 0 or len(mask4[1]) < 6:
            return None
        mask = []
        for x, y in zip(mask4[0], mask4[1]):
            mask.append(max(0, min(int(x), size)))
            mask.append(max(0, min(int(y), size)))
        masks.append([mask])
        label.append(box4[0])
        boxes.append(box4[1:5])
    if not len(boxes):
        return None
    # del copied results
    del results4
    if len(boxes) == len(label) == len(masks):
        random_hsv(image4)
        label = numpy.array(label, dtype=numpy.int64)
        boxes = numpy.array(boxes, dtype=numpy.float32)
        return dict(filename=filename, image=image4, label=label, boxes=boxes, masks=masks)
    else:
        return None


def mix_up(self, index1, index2):
    size = numpy.random.choice(self.image_sizes)

    data1 = mosaic(self, index1, size)
    data2 = mosaic(self, index2, size)
    alpha = numpy.random.beta(32.0, 32.0)

    if data1 is not None and data2 is not None:
        image1 = data1['image']
        label1 = data1['label']
        boxes1 = data1['boxes']
        masks1 = data1['masks']

        image2 = data2['image']
        label2 = data2['label']
        boxes2 = data2['boxes']
        masks2 = data2['masks']

        image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
        boxes = numpy.concatenate((boxes1, boxes2), 0)
        label = numpy.concatenate((label1, label2), 0)
        masks1.extend(masks2)

        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes, masks=masks1)
    if data1 is None and data2 is not None:
        image = data2['image']
        label = data2['label']
        boxes = data2['boxes']
        masks = data2['masks']

        return dict(filename=data2['filename'], image=image, label=label, boxes=boxes, masks=masks)
    if data1 is not None and data2 is None:
        image = data1['image']
        label = data1['label']
        boxes = data1['boxes']
        masks = data1['masks']

        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes, masks=masks)
    return None


def process(self, data):
    image = data['image']
    label = data['label']
    boxes = data['boxes']
    masks = data['masks']

    shape = image.shape

    results = dict()
    results['filename'] = data['filename']
    results['img_info'] = {'height': shape[0], 'width': shape[1]}
    results['ann_info'] = {'labels': label, 'bboxes': boxes, 'masks': masks}
    results['bbox_fields'] = []
    results['mask_fields'] = []
    results['ori_filename'] = basename(data['filename'])
    results['img'] = image
    results['img_fields'] = ['img']
    results['img_shape'] = shape
    results['ori_shape'] = shape
    results['pad_shape'] = shape

    results['scale_factor'] = numpy.array([1, 1, 1, 1], dtype=numpy.float32)
    return self.pipeline(results)


def box2field():
    box2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore': 'gt_masks_ignore'}
    box2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore': 'gt_labels_ignore'}
    return box2label, box2mask


class Equalize:
    def __call__(self, results, _):
        image = results['img']
        image = ImageOps.equalize(image)

        results['img'] = image
        return results


class Invert:
    def __call__(self, results, _):
        image = results['img']
        image = ImageOps.invert(image)

        results['img'] = image
        return results


class Identity:
    def __call__(self, results, _):
        return results


class Normalize:
    def __call__(self, results, _):
        image = results['img']
        image = ImageOps.autocontrast(image)

        results['img'] = image
        return results


class Brightness:
    def __call__(self, results, magnitude):
        image = results['img']
        if random.random() > 0.5:
            magnitude = (magnitude / max_value) * 1.8 + 0.1
            results['img'] = ImageEnhance.Brightness(image).enhance(magnitude)
            return results
        else:
            magnitude = (magnitude / max_value) * 0.9

            if random.random() > 0.5:
                magnitude *= -1

            results['img'] = ImageEnhance.Brightness(image).enhance(magnitude)
            return results


class Color:
    def __call__(self, results, magnitude):
        image = results['img']
        if random.random() > 0.5:
            magnitude = (magnitude / max_value) * 1.8 + 0.1
            results['img'] = ImageEnhance.Color(image).enhance(magnitude)
            return results
        else:
            magnitude = (magnitude / max_value) * 0.9

            if random.random() > 0.5:
                magnitude *= -1

            results['img'] = ImageEnhance.Color(image).enhance(magnitude)
            return results


class Contrast:
    def __call__(self, results, magnitude):
        image = results['img']
        if random.random() > 0.5:
            magnitude = (magnitude / max_value) * 1.8 + 0.1
            results['img'] = ImageEnhance.Contrast(image).enhance(magnitude)
            return results
        else:
            magnitude = (magnitude / max_value) * 0.9

            if random.random() > 0.5:
                magnitude *= -1

            results['img'] = ImageEnhance.Contrast(image).enhance(magnitude)
            return results


class Sharpness:
    def __call__(self, results, magnitude):
        image = results['img']
        if random.random() > 0.5:
            magnitude = (magnitude / max_value) * 1.8 + 0.1
            results['img'] = ImageEnhance.Sharpness(image).enhance(magnitude)
            return results
        else:
            magnitude = (magnitude / max_value) * 0.9

            if random.random() > 0.5:
                magnitude *= -1

            results['img'] = ImageEnhance.Sharpness(image).enhance(magnitude)
            return results


class Solar:
    def __call__(self, results, magnitude):
        image = results['img']

        magnitude = int((magnitude / max_value) * 256)
        if random.random() > 0.5:
            results['img'] = ImageOps.solarize(image, magnitude)
        else:
            results['img'] = ImageOps.solarize(image, 256 - magnitude)
        return results


class Poster:
    def __call__(self, results, magnitude):
        image = results['img']

        magnitude = int((magnitude / max_value) * 4)
        if random.random() > 0.5:
            if magnitude >= 8:
                return results
            results['img'] = ImageOps.posterize(image, magnitude)
            return results
        else:
            if random.random() > 0.5:
                magnitude = 4 - magnitude
            else:
                magnitude = 4 + magnitude

            if magnitude >= 8:
                return results
            results['img'] = ImageOps.posterize(image, magnitude)
            return results


class Rotate:
    @staticmethod
    def _rotate_image(results, angle, center, scale):
        image = results['img']
        image = numpy.asarray(image)
        results['img'] = mmcv.rgb2bgr(image)

        for key in results.get('img_fields', ['img']):
            img = results[key].copy()

            img_rotated = mmcv.imrotate(img, angle, center, scale)
            img_rotated = img_rotated.astype(img.dtype)
            img_rotated = mmcv.bgr2rgb(img_rotated)

            results[key] = Image.fromarray(img_rotated)

    @staticmethod
    def _rotate_boxes(results, rotate_matrix):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])
            coordinates = numpy.concatenate((coordinates,
                                             numpy.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                                            axis=1)
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = numpy.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x = numpy.min(rotated_coords[:, :, 0], axis=1)
            min_y = numpy.min(rotated_coords[:, :, 1], axis=1)
            max_x = numpy.max(rotated_coords[:, :, 0], axis=1)
            max_y = numpy.max(rotated_coords[:, :, 1], axis=1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _rotate_masks(results, angle, center, scale):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, 0)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results, magnitude):
        size = results['img'].size

        magnitude = int((magnitude / max_value) * 90)
        if random.random() > 0.5:
            magnitude *= -1

        center = ((size[0] - 1) * 0.5, (size[1] - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -magnitude, 1)

        self._rotate_image(results, magnitude, center, 1)
        self._rotate_boxes(results, matrix)
        self._rotate_masks(results, magnitude, center, 1)
        self._filter_invalid(results)
        return results


class ShearX:
    @staticmethod
    def _shear_image(results, magnitude, direction):
        image = results['img']
        image = numpy.asarray(image)
        results['img'] = mmcv.rgb2bgr(image)

        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(img, magnitude, direction)
            img_sheared = img_sheared.astype(img.dtype)
            img_sheared = mmcv.bgr2rgb(img_sheared)

            results[key] = Image.fromarray(img_sheared)

    @staticmethod
    def _shear_boxes(results, magnitude, direction):
        h, w, c = results['img_shape']
        if direction == 'horizontal':
            shear_matrix = numpy.stack([[1, magnitude], [0, 1]]).astype(numpy.float32)  # [2, 2]
        else:
            shear_matrix = numpy.stack([[1, 0], [magnitude, 1]]).astype(numpy.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose((2, 1, 0)).astype(numpy.float32)  # [nb_box, 2, 4]
            new_coords = numpy.matmul(shear_matrix[None, :, :], coordinates)  # [nb_box, 2, 4]
            min_x = numpy.min(new_coords[:, 0, :], axis=-1)
            min_y = numpy.min(new_coords[:, 1, :], axis=-1)
            max_x = numpy.max(new_coords[:, 0, :], axis=-1)
            max_y = numpy.max(new_coords[:, 1, :], axis=-1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _shear_masks(results, magnitude, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results, magnitude):
        magnitude = (magnitude / max_value) * 0.3
        if random.random() > 0.5:
            magnitude *= -1

        self._shear_image(results, magnitude, 'horizontal')
        self._shear_boxes(results, magnitude, 'horizontal')
        self._shear_masks(results, magnitude, 'horizontal')
        self._filter_invalid(results)
        return results


class ShearY:
    @staticmethod
    def _shear_image(results, magnitude, direction):
        image = results['img']
        image = numpy.asarray(image)
        results['img'] = mmcv.rgb2bgr(image)

        for key in results.get('img_fields', ['img']):
            img = results[key]

            img_sheared = mmcv.imshear(img, magnitude, direction)
            img_sheared = img_sheared.astype(img.dtype)
            img_sheared = mmcv.bgr2rgb(img_sheared)

            results[key] = Image.fromarray(img_sheared)

    @staticmethod
    def _shear_boxes(results, magnitude, direction):
        h, w, c = results['img_shape']
        if direction == 'horizontal':
            shear_matrix = numpy.stack([[1, magnitude], [0, 1]]).astype(numpy.float32)  # [2, 2]
        else:
            shear_matrix = numpy.stack([[1, 0], [magnitude, 1]]).astype(numpy.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose((2, 1, 0)).astype(numpy.float32)  # [nb_box, 2, 4]
            new_coords = numpy.matmul(shear_matrix[None, :, :], coordinates)  # [nb_box, 2, 4]
            min_x = numpy.min(new_coords[:, 0, :], axis=-1)
            min_y = numpy.min(new_coords[:, 1, :], axis=-1)
            max_x = numpy.max(new_coords[:, 0, :], axis=-1)
            max_y = numpy.max(new_coords[:, 1, :], axis=-1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _shear_masks(results, magnitude, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results, magnitude):
        magnitude = (magnitude / max_value) * 0.3
        if random.random() > 0.5:
            magnitude *= -1

        self._shear_image(results, magnitude, 'vertical')
        self._shear_boxes(results, magnitude, 'vertical')
        self._shear_masks(results, magnitude, 'vertical')
        self._filter_invalid(results)

        return results


class TranslateX:
    @staticmethod
    def _translate_image(results, offset, direction):
        image = results['img']
        image = numpy.asarray(image)
        results['img'] = mmcv.rgb2bgr(image)

        for key in results.get('img_fields', ['img']):
            img = results[key].copy()

            img_translated = mmcv.imtranslate(img, offset, direction)
            img_translated = img_translated.astype(img.dtype)
            img_translated = mmcv.bgr2rgb(img_translated)

            results[key] = Image.fromarray(img_translated)

    @staticmethod
    def _translate_boxes(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            if direction == 'horizontal':
                min_x = numpy.maximum(0, min_x + offset)
                max_x = numpy.minimum(w, max_x + offset)
            elif direction == 'vertical':
                min_y = numpy.maximum(0, min_y + offset)
                max_y = numpy.minimum(h, max_y + offset)

            results[key] = numpy.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    @staticmethod
    def _translate_masks(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, 0)

    @staticmethod
    def _filter_invalid(results):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > 0) & (bbox_h > 0)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]
        return results

    def __call__(self, results, magnitude):
        magnitude = (magnitude / max_value) * 0.5
        if random.random() > 0.5:
            magnitude *= -1

        offset = magnitude * min(results['img'].size)

        self._translate_image(results, offset, 'horizontal')
        self._translate_boxes(results, offset, 'horizontal')
        self._translate_masks(results, offset, 'horizontal')
        self._filter_invalid(results)
        return results


class TranslateY:
    @staticmethod
    def _translate_image(results, offset, direction):
        image = results['img']
        image = numpy.asarray(image)
        results['img'] = mmcv.rgb2bgr(image)

        for key in results.get('img_fields', ['img']):
            img = results[key].copy()

            img_translated = mmcv.imtranslate(img, offset, direction)
            img_translated = img_translated.astype(img.dtype)
            img_translated = mmcv.bgr2rgb(img_translated)

            results[key] = Image.fromarray(img_translated)

    @staticmethod
    def _translate_boxes(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            if direction == 'horizontal':
                min_x = numpy.maximum(0, min_x + offset)
                max_x = numpy.minimum(w, max_x + offset)
            elif direction == 'vertical':
                min_y = numpy.maximum(0, min_y + offset)
                max_y = numpy.minimum(h, max_y + offset)

            results[key] = numpy.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    @staticmethod
    def _translate_masks(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, 0)

    @staticmethod
    def _filter_invalid(results):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > 0) & (bbox_h > 0)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]
        return results

    def __call__(self, results, magnitude):
        magnitude = (magnitude / max_value) * 0.5
        if random.random() > 0.5:
            magnitude *= -1

        offset = magnitude * min(results['img'].size)

        self._translate_image(results, offset, 'vertical')
        self._translate_boxes(results, offset, 'vertical')
        self._translate_masks(results, offset, 'vertical')
        self._filter_invalid(results)
        return results


@PIPELINES.register_module()
class RandomAugment:
    def __init__(self, mean=9, sigma=0.5, n=2):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (Equalize(), Identity(), Invert(), Normalize(),
                          Rotate(), ShearX(), ShearY(), TranslateX(), TranslateY(),
                          Brightness(), Color(), Contrast(), Sharpness(), Solar(), Poster())

    def __call__(self, results):
        image = results['img']
        image = mmcv.bgr2rgb(image)
        image = Image.fromarray(image)

        results['img'] = image
        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))

            results = transform(results, magnitude)

        image = results['img']
        image = mmcv.rgb2bgr(numpy.asarray(image))

        results['img'] = image
        return results


@PIPELINES.register_module()
class GridMask:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results

        image = results['img']
        shape = results['img_shape'][:2]

        h = int(shape[0] * 1.5)
        w = int(shape[1] * 1.5)
        d = numpy.random.randint(2, min(shape))

        st_h = numpy.random.randint(d)
        st_w = numpy.random.randint(d)
        mask = numpy.ones((h, w), numpy.float32)

        for i in range(h // d):
            s = d * i + st_h
            t = min(s + min(max(int(d / 2 + 0.5), 1), d - 1), h)
            mask[s:t, :] *= 0

        for i in range(w // d):
            s = d * i + st_w
            t = min(s + min(max(int(d / 2 + 0.5), 1), d - 1), w)
            mask[:, s:t] *= 0

        delta_h = (h - shape[0]) // 2
        delta_w = (w - shape[1]) // 2

        mask = mask[delta_h:delta_h + shape[0], delta_w:delta_w + shape[1]]

        mask = 1 - mask.astype(numpy.float32)
        mask = numpy.expand_dims(mask, 2).repeat(3, axis=2)

        results['img'] = (image * mask).astype('uint8')
        return results
