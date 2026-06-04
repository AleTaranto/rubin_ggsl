"""
Critical lines, caustics, and related singularities.

Critical lines are where the magnification diverges (det(A) = 0).
Caustics are the images of critical lines in source plane.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import find_objects, label


class CriticalLines:
    """
    Identify and analyze critical lines and caustics.
    """
    
    def __init__(self, magnification_calc):
        """
        Initialize critical line finder.
        
        Parameters
        ----------
        magnification_calc : Magnification
            Magnification calculator
        """
        self.mag_calc = magnification_calc
        self.deflection = magnification_calc.deflection
        self.cluster = magnification_calc.cluster
    
    def find_critical_lines(self, x_grid, y_grid, critical_threshold=1e3):
        """
        Find critical lines on a grid.
        
        Critical lines occur where magnification -> infinity (det(A) ~ 0).
        
        Parameters
        ----------
        x_grid, y_grid : ndarray
            1D arrays defining grid
        critical_threshold : float
            Magnification threshold for detecting critical regions
        
        Returns
        -------
        critical_mask : ndarray of shape (len(y_grid), len(x_grid))
            Boolean mask indicating critical regions
        magnification_grid : ndarray
            Magnification on grid
        """
        # Compute magnification on grid
        mu_grid = self.mag_calc.compute_magnification_grid(x_grid, y_grid)
        
        # Clamp magnification to reasonable range for visualization
        mu_clipped = np.clip(mu_grid, -critical_threshold, critical_threshold)
        
        # Critical regions: where mu is very large
        critical_mask = np.abs(mu_clipped) > critical_threshold * 0.9
        
        return critical_mask, mu_grid
    
    def find_caustics(self, x_grid, y_grid, source_plane=False):
        """
        Find caustics (images of critical lines in source plane).
        
        For each point on the image plane, compute its position in source plane
        using the lens equation: beta = theta - alpha(theta)
        
        Then trace which image-plane points map to the same source-plane region.
        
        Parameters
        ----------
        x_grid, y_grid : ndarray
            1D arrays defining grid
        source_plane : bool
            If True, return caustics in source plane; otherwise image plane
        
        Returns
        -------
        caustics : list of ndarray
            List of caustic curves (each a 2D array of points)
        """
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute deflection
        alpha_x, alpha_y = self.deflection.deflection_angle(X, Y)
        
        # Source plane positions: beta = theta - alpha
        beta_x = X - alpha_x
        beta_y = Y - alpha_y
        
        # Find where Jacobian = 0 (critical points)
        A11, A12, A21, A22 = self.mag_calc.magnification_tensor(X, Y)
        det_A = A11 * A22 - A12 * A21
        
        # Trace critical lines in source plane
        critical_in_source = np.abs(det_A) < np.percentile(np.abs(det_A), 1)
        
        # Find connected components
        labeled_array, num_features = label(critical_in_source)
        
        caustics = []
        for feature_id in range(1, num_features + 1):
            mask = labeled_array == feature_id
            indices = np.where(mask)
            
            if len(indices[0]) > 10:  # Only significant caustics
                if source_plane:
                    caustic_x = beta_x[mask]
                    caustic_y = beta_y[mask]
                else:
                    caustic_x = X[mask]
                    caustic_y = Y[mask]
                
                caustics.append(np.column_stack([caustic_x, caustic_y]))
        
        return caustics
    
    def characterize_critical_line(self, caustic_points):
        """
        Characterize a critical line (or caustic).
        
        Parameters
        ----------
        caustic_points : ndarray of shape (n_points, 2)
            Points on the critical line/caustic
        
        Returns
        -------
        properties : dict
            Dictionary with properties:
            - length: Total path length
            - curvature: Mean curvature
            - extent: Bounding box size
        """
        if len(caustic_points) < 3:
            return {}
        
        # Compute path length
        diffs = np.diff(caustic_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        length = np.sum(distances)
        
        # Compute bounding extent
        extent = np.ptp(caustic_points, axis=0)  # Peak-to-peak (max-min)
        
        # Simple curvature estimate
        if len(caustic_points) > 3:
            # Fit circle to each triplet and average curvature
            curvatures = []
            for i in range(1, len(caustic_points) - 1):
                p1 = caustic_points[i-1]
                p2 = caustic_points[i]
                p3 = caustic_points[i+1]
                
                # Curvature from three points
                d12 = np.linalg.norm(p2 - p1)
                d23 = np.linalg.norm(p3 - p2)
                d13 = np.linalg.norm(p3 - p1)
                
                if d12 > 0 and d23 > 0:
                    # Using formula: curvature ~ 2 * area / (d12 * d23 * d13)
                    area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                       (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area > 0:
                        curv = 2 * area / (d12 * d23)
                        curvatures.append(curv)
            
            curvature = np.mean(curvatures) if curvatures else 0.0
        else:
            curvature = 0.0
        
        return {
            'length': length,
            'extent': extent,
            'curvature': curvature,
            'n_points': len(caustic_points),
        }


class MultiplicityPredictor:
    """
    Predict multiple image properties from critical line topology.
    """
    
    def __init__(self, critical_lines_calc):
        """
        Initialize multiplicity predictor.
        
        Parameters
        ----------
        critical_lines_calc : CriticalLines
            Critical lines calculator
        """
        self.crit_calc = critical_lines_calc
    
    def predict_n_images(self, source_position, x_grid, y_grid):
        """
        Predict number of images for a source at given position.
        
        Uses winding number or critical line topology.
        
        Parameters
        ----------
        source_position : tuple
            (x, y) position in source plane
        x_grid, y_grid : ndarray
            Grid for computation
        
        Returns
        -------
        n_images : int
            Predicted number of images
        """
        # Find caustics
        caustics = self.crit_calc.find_caustics(x_grid, y_grid, source_plane=True)
        
        # Count how many critical lines (caustics) contain the source
        # Simplified: count caustics within some radius of source
        source_x, source_y = source_position
        n_enclosed = 0
        
        for caustic in caustics:
            # Rough check: if source is within bounding box of caustic
            if (np.min(caustic[:, 0]) <= source_x <= np.max(caustic[:, 0]) and
                np.min(caustic[:, 1]) <= source_y <= np.max(caustic[:, 1])):
                n_enclosed += 1
        
        # Heuristic: n_images = 2 * n_enclosed + 1
        n_images = 2 * n_enclosed + 1
        return max(n_images, 1)
    
    def magnification_distribution(self, source_position, x_grid, y_grid):
        """
        Compute magnifications of all images for a source.
        
        Parameters
        ----------
        source_position : tuple
            (x, y) position in source plane
        x_grid, y_grid : ndarray
            Grid for computation
        
        Returns
        -------
        magnifications : ndarray
            Array of image magnifications
        """
        # This is a simplified version
        # Full implementation would trace rays back from each image
        
        # Placeholder: return typical magnification distribution
        n_images = self.predict_n_images(source_position, x_grid, y_grid)
        
        # Typical: one bright central image + multiple fainter outer images
        mags = [10.0]  # Central image
        for i in range(n_images - 1):
            mags.append(2.0 + i)  # Dimmer images
        
        return np.array(mags)
