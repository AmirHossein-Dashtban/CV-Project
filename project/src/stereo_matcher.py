import numpy as np
import cv2


class StereoMatcher:
    def __init__(self, window_size=5, max_disparity=64, cost_metric="SAD"):
        """
        Stereo Matching implementation using SAD, SSD or NCC.
        window_size: odd integer.
        max_disparity: integer.
        cost_metric: SAD, SSD, NCC.
        """
        self.window_size = window_size
        self.max_disparity = max_disparity
        self.cost_metric = cost_metric

    def compute_disparity(self, img_left, img_right):
        """
        Computes disparity map using winner-take-all block matching.
        """
        # Convert to grayscale if necessary
        if len(img_left.shape) == 3:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)

        h, w = img_left.shape
        disparity_map = np.zeros((h, w), dtype=np.float32)
        min_costs = np.full((h, w), np.inf, dtype=np.float32)

        # For SAD/SSD, we want to minimize cost.
        # For NCC, we want to maximize correlation.
        if self.cost_metric == "NCC":
            min_costs = np.full((h, w), -np.inf, dtype=np.float32)
            n = self.window_size * self.window_size

            # Precompute L sums
            sum_l = cv2.boxFilter(
                img_left, -1, (self.window_size, self.window_size), normalize=False
            )
            sum_l2 = cv2.boxFilter(
                img_left**2, -1, (self.window_size, self.window_size), normalize=False
            )
            den_l = sum_l2 - (sum_l**2 / n)

            # Precompute R sums (will be shifted inside loop)
            sum_r_base = cv2.boxFilter(
                img_right, -1, (self.window_size, self.window_size), normalize=False
            )
            sum_r2_base = cv2.boxFilter(
                img_right**2, -1, (self.window_size, self.window_size), normalize=False
            )

        for d in range(self.max_disparity):
            # Shift Right Image by 'd' to the right
            shifted_right = np.zeros_like(img_right)
            if d > 0:
                shifted_right[:, d:] = img_right[:, :-d]
            else:
                shifted_right = img_right.copy()

            if self.cost_metric == "NCC":
                # Shift precomputed sums for efficiency
                sum_r = np.zeros_like(sum_r_base)
                sum_r2 = np.zeros_like(sum_r2_base)
                if d > 0:
                    sum_r[:, d:] = sum_r_base[:, :-d]
                    sum_r2[:, d:] = sum_r2_base[:, :-d]
                else:
                    sum_r = sum_r_base.copy()
                    sum_r2 = sum_r2_base.copy()

                sum_lr = cv2.boxFilter(
                    img_left * shifted_right,
                    -1,
                    (self.window_size, self.window_size),
                    normalize=False,
                )

                num = sum_lr - (sum_l * sum_r / n)
                den_r = sum_r2 - (sum_r**2 / n)

                denom = np.sqrt(np.maximum(den_l * den_r, 1e-6))
                cost = num / denom

                mask = (cost > min_costs) & (denom > 1e-3)
                disparity_map[mask] = d
                min_costs[mask] = cost[mask]

            else:
                # Compute pixel-wise error
                if self.cost_metric == "SAD":
                    diff = np.abs(img_left - shifted_right)
                elif self.cost_metric == "SSD":
                    diff = (img_left - shifted_right) ** 2
                else:
                    diff = np.abs(img_left - shifted_right)

                # Sum of difference over a window
                cost = cv2.boxFilter(
                    diff, -1, (self.window_size, self.window_size), normalize=False
                )

                mask = cost < min_costs
                disparity_map[mask] = d
                min_costs[mask] = cost[mask]

        return disparity_map


def compute_lr_disparity(
    img_left, img_right, window_size=5, max_disparity=64, cost_metric="SAD"
):
    """Computes left and right disparity for L-R check."""
    matcher = StereoMatcher(
        window_size=window_size, max_disparity=max_disparity, cost_metric=cost_metric
    )

    # Left disparity: searching right image for match in left
    # We shift right image by d to position (x-d)
    disp_left = matcher.compute_disparity(img_left, img_right)

    # Right disparity: searching left image for match in right
    # To use the same matcher, we flip images and then flip result
    img_left_flip = cv2.flip(img_left, 1)
    img_right_flip = cv2.flip(img_right, 1)
    disp_right_flip = matcher.compute_disparity(img_right_flip, img_left_flip)
    disp_right = cv2.flip(disp_right_flip, 1)

    return disp_left, disp_right
