import numpy as np
class PatchPreds:
    def __init__(self,
                 patch_height=256,
                 patch_width=256,
                 width_stride=200,
                 height_stride=25,
                 margin_width=20,
                 margin_weights=0.3
                 ):
        self.patch_height_ = patch_height
        self.patch_width_ = patch_width
        self.width_stride_ = width_stride
        self.height_stride_ = height_stride
        self.margin_width_ = margin_width
        self.margin_weights_ = margin_weights

    def predict(self, image:np.array, model=None):
        ori_img_height, ori_img_width, _ = image.shape

        # pad to the below and pad to the right
        image = self.image_padding(image)
        img_height, img_width, _ = image.shape

        # 1. get all patches
        patch_indices = self.get_all_patch_indices(img_height, img_width)

        # 2. calculate weights
        weights_array = self.get_weights_array(img_height, img_width, patch_indices)

        # 3. make all patch_indices as a batch
        patches, weights = self.get_all_patches(image, weights_array, patch_indices=patch_indices)

        # 4. make prediction
        preds = model(patches)
        preds = preds.numpy()
        preds_weighted = preds * weights

        # 5. reconstruct to original image
        preds = self.combine_patches(
            patch_preds=preds_weighted,
            patch_indices=patch_indices,
            img_height=img_height,
            img_width=img_width)

        return preds[:ori_img_height, :ori_img_width]



    def get_all_patch_indices(self, img_height, img_width):
        """

        Parameters
        ----------
        img_height
        img_width

        Returns
        list of each patch's upper left position
        [(pos_height, pos_width)]
        -------

        """
        res = []
        #(height_left_upper, width_left_upper)
        h = 0
        # loop for height
        while True:
            w = 0
            # loop for height
            while True:
                res.append((h, w))
                if w + self.patch_width_ == img_width:
                    break
                else:
                    w = w + self.width_stride_
                    w = min(w, img_width - self.patch_width_)
            if h + self.patch_height_ == img_height:
                break
            else:
                h = h + self.height_stride_
                h = min(h, img_height - self.patch_height_)
        return res

    def get_single_patch_weights(self):
        array = np.ones(shape=(self.patch_height_, self.patch_width_)) * self.margin_weights_
        array[
        self.margin_width_ : self.patch_height_ - self.margin_width_,
        self.margin_width_ : self.patch_width_ - self.margin_width_
        ] = 1.0
        return array

    def get_weights_array(self, img_height, img_width, patch_indices):
        weights = np.zeros(shape=(img_height, img_width))
        for patch in patch_indices:
            h, w = patch
            weights[h:h+self.patch_height_, w:w+self.patch_width_] += self.get_single_patch_weights()
        return weights

    def get_all_patches(self, image: np.array, weights:np.array, patch_indices):
        res_img = []
        res_weight = []
        for patch in patch_indices:
            h, w = patch
            res_img.append(image[h: h+self.patch_height_, w: w+self.patch_width_])
            res_weight.append( self.get_single_patch_weights() /
                               weights[h: h+self.patch_height_, w: w+self.patch_width_])
        return np.stack(res_img, axis=0), np.stack(res_weight, axis=0)

    def image_padding(self, img):
        h, w, _ = img.shape
        h_pad = max(0, self.patch_height_ - h)
        w_pad = max(0, self.patch_width_ - w)
        return np.pad(img, ((0, h_pad), (0, w_pad), (0,0)), mode="constant", constant_values=0.0)

    def combine_patches(self, patch_preds, patch_indices, img_height, img_width):
        array = np.zeros(shape=(img_height, img_width))
        for indices, preds in zip(patch_indices, patch_preds):
            h, w = indices
            array[h:h+self.patch_height_, w:w+self.patch_width_] += preds
        return array





