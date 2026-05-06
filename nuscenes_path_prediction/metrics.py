import torch


class PathForecastingMetrics:
    """
    A collection of metrics designed to evaluate generative path forecasting models.
    Updated to include Hausdorff Distance and Path Plausibility as suggested
    to handle subset/partial path issues.
    """

    @staticmethod
    def hausdorff_distance_components(
        pred_path: torch.Tensor,
        gt_path: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bidirectional Hausdorff distance and its directional max errors (same pairing as ``max(prec, rec)``).

        Returns:
            Tuple ``(hausdorff_dist, precision_max_err, recall_max_err)`` as scalar tensors.
        """
        if pred_path.numel() == 0 or gt_path.numel() == 0:
            inf = torch.tensor(float("inf"), device=pred_path.device)
            return inf, inf, inf

        distances = torch.cdist(pred_path.float(), gt_path.float(), p=2.0)

        precision_max_err = torch.min(distances, dim=1).values.max()
        recall_max_err = torch.min(distances, dim=0).values.max()
        hausdorff_dist = torch.max(precision_max_err, recall_max_err)
        return hausdorff_dist, precision_max_err, recall_max_err

    @staticmethod
    def calculate_hausdorff_distance(pred_path: torch.Tensor, gt_path: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates the Bidirectional Hausdorff Distance between two paths.

        This metric is more sensitive to structural failures than ADE because it
        looks at the 'worst-case' disagreement. It captures two critical aspects:
        1. Precision (Max error from Pred to GT): Does the prediction stay on the path?
        2. Recall (Max error from GT to Pred): Does the prediction cover the entire GT?

        Formula: $d_H(P, G) = \max(\max_{p \in P} \min_{g \in G} \|p-g\|, \max_{g \in G} \min_{p \in P} \|g-p\|)$

        Args:
            pred_path: Tensor of shape (L_pred, 2) containing (y, x) coordinates.
            gt_path: Tensor of shape (L_gt, 2) containing (y, x) coordinates.

        Returns:
            hausdorff_dist: Scalar tensor representing the maximum deviation.
        """
        hd, _, _ = PathForecastingMetrics.hausdorff_distance_components(pred_path, gt_path)
        return hd

    @staticmethod
    def calculate_hausdorff_distance_p90(pred_path: torch.Tensor, gt_path: torch.Tensor) -> torch.Tensor:
        """
        Calculates the 90th percentile (p90) bidirectional Hausdorff distance between two paths.

        Unlike the classic Hausdorff distance (which considers worst-case outlier errors),
        the p90 version returns the 90th percentile error in both directions:

        - For each predicted point, computes the minimum distance to any ground truth point ("precision").
        - For each ground truth point, computes the minimum distance to any predicted point ("recall").
        - Reports the maximum of the 90th percentile of both sets of distances.

        This gives a more robust distance metric, less sensitive to outlier points or partial path matches.

        Args:
            pred_path: Tensor of shape (L_pred, 2) containing (y, x) coordinates.
            gt_path: Tensor of shape (L_gt, 2) containing (y, x) coordinates.

        Returns:
            Scalar tensor representing the 90th percentile bidirectional Hausdorff distance.
        """

        if pred_path.numel() == 0 or gt_path.numel() == 0:
            return torch.tensor(float('inf'), device=pred_path.device)

        # Compute pairwise distance matrix (L_pred, L_gt)
        distances = torch.cdist(pred_path.float(), gt_path.float(), p=2.0)

        # Precision component: For each pred point, find the min distance to GT, then take the MAX
        # High value means the model predicted a point far away from any valid ground truth.
        precision_err = torch.min(distances, dim=1).values
        hd90_pred = torch.quantile(precision_err, 0.9)

        # Recall component: For each GT point, find the min distance to Pred, then take the MAX
        # High value means a part of the ground truth was completely ignored (e.g., partial path).
        recall_err = torch.min(distances, dim=0).values
        hd90_gt = torch.quantile(recall_err, 0.9)

        # The Hausdorff distance is the maximum of these two directional errors
        return torch.max(hd90_pred, hd90_gt)

    @staticmethod
    def calculate_bidirectional_ade(pred_path: torch.Tensor, gt_path: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Bidirectional Spatial Average Displacement Error (Bi-ADE).

        This metric averages:
        1. Pred → GT (precision-like): Are predicted points close to GT?
        2. GT → Pred (recall-like): Is the GT fully covered by prediction?

        Args:
            pred_path: Tensor of shape (L_pred, 2)
            gt_path: Tensor of shape (L_gt, 2)

        Returns:
            bi_ade: Scalar tensor
        """
        if pred_path.numel() == 0 or gt_path.numel() == 0:
            return torch.tensor(float('inf'), device=pred_path.device)

        distances = torch.cdist(pred_path.float(), gt_path.float(), p=2.0)

        # pred → gt
        min_pred_to_gt = torch.min(distances, dim=1).values
        ade_pred_to_gt = torch.mean(min_pred_to_gt)

        # gt → pred
        min_gt_to_pred = torch.min(distances, dim=0).values
        ade_gt_to_pred = torch.mean(min_gt_to_pred)

        # symmetric average
        bi_ade = 0.5 * (ade_pred_to_gt + ade_gt_to_pred)

        return bi_ade

    @staticmethod
    def min_ade(pred_path: torch.Tensor, gt_paths: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the Minimum ADE across all possible valid GT paths.
        This metric rewards the model for committing to ONE valid mode,
        and heavily penalizes models that output the "average" of two modes.

        Args:
            pred_path: Tensor of shape (L_pred, 2) representing the model's single prediction.
            gt_paths: List of M Tensors, each of shape (L_m, 2), representing all valid GT options.

        Returns:
            min_ade_val: The ADE to the closest GT path.
        """
        if not gt_paths:
            return torch.tensor(float('inf'), device=pred_path.device)

        ades = []
        for gt_path in gt_paths:
            ade = PathForecastingMetrics.calculate_bidirectional_ade(pred_path, gt_path)
            ades.append(ade)

        # Return the minimum error across all possible valid paths
        min_ade_val = torch.stack(ades).min()
        return min_ade_val

    @staticmethod
    def min_hd(pred_path: torch.Tensor, gt_paths: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the minimum Hausdorff distance across all possible valid GT paths.
        """
        if not gt_paths:
            return torch.tensor(float('inf'), device=pred_path.device)

        hds = []
        for gt_path in gt_paths:
            hd = PathForecastingMetrics.calculate_hausdorff_distance(pred_path, gt_path)
            hds.append(hd)
        return torch.stack(hds).min()

    @staticmethod
    def min_hd_components(
        pred_path: torch.Tensor,
        gt_paths: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Minimum Hausdorff distance over valid GT paths, with precision/recall max errors
        taken from the same GT path that achieves that minimum (argmin over modes).
        """
        if not gt_paths:
            inf = torch.tensor(float("inf"), device=pred_path.device)
            return inf, inf, inf

        hds: list[torch.Tensor] = []
        precs: list[torch.Tensor] = []
        recalls: list[torch.Tensor] = []
        for gt_path in gt_paths:
            hd, p, r = PathForecastingMetrics.hausdorff_distance_components(pred_path, gt_path)
            hds.append(hd)
            precs.append(p)
            recalls.append(r)
        stack_hd = torch.stack(hds)
        k_star = int(stack_hd.argmin().item())
        return stack_hd[k_star], precs[k_star], recalls[k_star]

    @staticmethod
    def min_hd_p90(pred_path: torch.Tensor, gt_paths: list[torch.Tensor]) -> torch.Tensor:
        """
        Minimum p90 Hausdorff distance across all valid GT paths (same reduction as min_hd).
        """
        if not gt_paths:
            return torch.tensor(float('inf'), device=pred_path.device)

        hds = []
        for gt_path in gt_paths:
            hd = PathForecastingMetrics.calculate_hausdorff_distance_p90(pred_path, gt_path)
            hds.append(hd)
        return torch.stack(hds).min()

    @staticmethod
    def off_road_rate(pred_path: torch.Tensor, drivable_area_map: torch.Tensor) -> torch.Tensor:
        """
        Calculates the percentage of the predicted path that falls outside the drivable area.
        Proves that the model's predictions are physically valid and context-aware.

        Args:
            pred_path: Tensor of shape (L_pred, 2) containing integer (y, x) grid coordinates.
            drivable_area_map: Binary tensor of shape (H, W) where 1 indicates drivable road
                               and 0 indicates off-road / obstacles.

        Returns:
            rate: Scalar tensor [0.0, 1.0] representing the fraction of off-road points.
        """
        if pred_path.numel() == 0:
            return torch.tensor(1.0, device=drivable_area_map.device)

        # Extract y and x coordinates. Ensure they are long (integers) for indexing
        y_coords = pred_path[:, 0].long()
        x_coords = pred_path[:, 1].long()

        H, W = drivable_area_map.shape

        # Create a mask for points that are strictly within the image boundaries
        valid_bounds = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)

        if not valid_bounds.any():
            return torch.tensor(1.0, device=drivable_area_map.device)  # Entirely out of bounds

        # Filter coordinates to prevent index out of bounds
        y_valid = y_coords[valid_bounds]
        x_valid = x_coords[valid_bounds]

        # Check map values at the valid coordinates (1 = road, 0 = off-road)
        path_validity = drivable_area_map[y_valid, x_valid]

        # Points that are 0 in the map are off-road
        off_road_points = (path_validity == 0).sum()

        # Add points that were completely out of image bounds as off-road
        out_of_bounds_count = (~valid_bounds).sum()
        total_off_road = off_road_points + out_of_bounds_count

        rate = total_off_road.float() / pred_path.shape[0]
        return rate
