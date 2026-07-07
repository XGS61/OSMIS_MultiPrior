import os
import torch
import warnings
from PIL import Image
from torchvision import transforms as TR
import torchvision.transforms.functional as F
from .recommended_config import get_recommended_config


def prepare_dataloading(opt):
    dataset = Dataset(opt)
    recommended_config = {"image resolution": dataset.image_resolution,
                          "noise_shape": dataset.recommended_config[0],
                          "num_blocks_g":  dataset.recommended_config[1],
                          "num_blocks_d":  dataset.recommended_config[2],
                          "num_blocks_d0": dataset.recommended_config[3],
                          "no_masks": dataset.no_masks,
                          "num_mask_channels": dataset.num_mask_channels}
    if not recommended_config["no_masks"] and not opt.no_masks:
        print("Using the training regime *with* segmentation masks")
    else:
        opt.no_masks = True
        print("Using the training regime *without* segmentation masks")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=opt.device != "cpu",
    )
    return dataloader, recommended_config


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        """
        The dataset class. Supports both regimes *with* and *without* segmentation masks.
        """
        self.device = opt.device
        # --- images --- #
        self.root_images = os.path.join(opt.dataroot, opt.dataset_name, "image")
        self.root_masks = os.path.join(opt.dataroot, opt.dataset_name, "mask")
        self.list_imgs = self.get_frames_list(self.root_images)
        assert len(self.list_imgs) > 0, "Found no images"
        self.image_resolution, self.recommended_config = get_recommended_config(self.get_im_resolution(opt.max_size))

        # --- masks --- #
        if os.path.isdir(self.root_masks) and not opt.no_masks:
            self.list_masks = self.get_frames_list(self.root_masks)
            assert len(self.list_imgs) == len(self.list_masks), \
                "Different number of images and masks %d vs %d" % (len(self.list_imgs), len(self.list_masks))
            for i in range(len(self.list_imgs)):
                assert os.path.splitext(self.list_imgs[i])[0] == os.path.splitext(self.list_masks[i])[0], \
                "Image and its mask must have same names %s - %s" % (self.list_imgs[i], self.list_masks[i])
            self.num_mask_channels = self.get_num_mask_channels()
            self.no_masks = False
        else:
            self.no_masks = True
            self.num_mask_channels = None

        print("Created a dataset of size =", len(self.list_imgs), "with image resolution", self.image_resolution)

    def get_frames_list(self, path):
        return sorted(os.listdir(path))

    def __len__(self):
        return 100000000  # so first epoch finishes only with break

    def get_im_resolution(self, max_size):
        """
        Iterate over images to determine image resolution.
        If there are images with different shapes, return the square of average size
        """
        res_list = list()
        for cur_img in self.list_imgs:
            img_pil = Image.open(os.path.join(self.root_images, cur_img)).convert("RGB")
            res_list.append(img_pil.size)
        all_res_equal = len(set(res_list)) <= 1
        if all_res_equal:
            size_1, size_2 = res_list[0]  # all images have same resolution -> using original resolution
        else:
            warnings.warn("Images in the dataset have different resolutions. Resizing them to squares of mean size.")
            size_1 = size_2 = sum([sum(item) for item in res_list]) / (2 * len(res_list))
        size_1, size_2 = self.bound_resolution(size_1, size_2, max_size)
        return size_2, size_1

    def bound_resolution(self, size_1, size_2, max_size):
        """
        Ensure the image shape does not exceed --max_size
        """
        if size_1 > max_size:
            size_1, size_2 = max_size, size_2 / (size_1 / max_size)
        if size_2 > max_size:
            size_1, size_2 = size_1 / (size_2 / max_size), max_size
        return int(size_1), int(size_2)

    def get_num_mask_channels(self):
        """
        Iterate over all masks to determine how many classes are there
        """
        max_index = 0
        for cur_mask in self.list_masks:
            im = TR.functional.to_tensor(Image.open(os.path.join(self.root_masks, cur_mask)))
            if (im.unique() * 256).max() > 30:
                # --- black-white map of one object and background --- #
                max_index = 2 if max_index < 2 else max_index
            else:
                # --- multiple semantic objects --- #
                cur_max = torch.max(torch.round(im * 256))
                max_index = cur_max + 1 if max_index < cur_max + 1 else max_index
        return int(max_index)

    def create_mask_channels(self, mask):
        """
        Convert a mask to one-hot representation
        """
        if (mask.unique() * 256).max() > 30:
            # --- only object and background--- #
            mask = torch.cat((1 - mask, mask), dim=0)
            return mask
        else:
            # --- multiple semantic objects --- #
            integers = torch.round(mask * 256)
            mask = torch.nn.functional.one_hot(integers.long(), num_classes=self.num_mask_channels)
            mask = mask.float()[0].permute(2, 0, 1)
            return mask

    def resize_with_pad(self, image, target_size, interpolation, fill):
        """Preserve anatomy aspect ratio and pad to the generator resolution."""
        target_h, target_w = target_size
        source_w, source_h = image.size
        scale = min(target_w / source_w, target_h / source_h)
        new_w = max(1, int(round(source_w * scale)))
        new_h = max(1, int(round(source_h * scale)))
        image = F.resize(image, size=(new_h, new_w), interpolation=interpolation)
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        return F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

    def __getitem__(self, index):
        output = dict()
        idx = index % len(self.list_imgs)
        target_size = self.image_resolution

        # --- image ---#
        img_pil = Image.open(os.path.join(self.root_images, self.list_imgs[idx])).convert("RGB")
        img_pil = self.resize_with_pad(
            img_pil, target_size, TR.InterpolationMode.BILINEAR, fill=0
        )
        img = F.to_tensor(img_pil)
        img = (img - 0.5) * 2
        output["images"] = img

        # --- mask ---#
        if not self.no_masks:
            mask_pil = Image.open(os.path.join(self.root_masks, self.list_imgs[idx][:-4] + ".png"))
            mask_pil = self.resize_with_pad(
                mask_pil, target_size, TR.InterpolationMode.NEAREST, fill=0
            )
            mask = F.to_tensor(mask_pil)
            mask = self.create_mask_channels(mask)  # mask should be N+1 channels
            output["masks"] = mask
            assert img.shape[1:] == mask.shape[1:], "Image and mask must have same dims %s" % (self.list_imgs[idx])
        return output



