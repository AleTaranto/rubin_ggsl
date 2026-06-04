"""
GGSL (Giant Gravitational Strong Lensing) observable definitions and event counters.

Defines what constitutes a detectable GGSL event and provides tools to
identify and characterize such events in simulated images.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class GGSLObservable:
    """
    Data class for a detected GGSL event.
    """
    event_id: int
    source_id: int
    n_images: int
    magnifications: np.ndarray  # Array of magnifications for each image
    total_magnification: float
    arc_length_arcsec: float
    arc_width_arcsec: float
    source_redshift: float
    lens_redshift: float
    Einstein_radius_arcsec: float = None
    caustic_properties: Dict = None
    
    def is_strong_arc(self, min_length_arcsec=1.0, min_mag=2.0):
        """Check if this is a strong arc (elongated image)."""
        elongation = self.arc_length_arcsec / max(self.arc_width_arcsec, 0.1)
        return (self.arc_length_arcsec > min_length_arcsec and 
                self.total_magnification > min_mag and 
                elongation > 5)
    
    def __repr__(self):
        return (f"GGSL(id={self.event_id}, n_img={self.n_images}, "
                f"mag={self.total_magnification:.1f}, len={self.arc_length_arcsec:.2f}\")")


class GGSLDefinitions:
    """
    Define criteria for what counts as a GGSL event.
    """
    
    def __init__(self, min_magnification=2.0, min_arc_length_arcsec=1.0,
                 min_image_separation_arcsec=0.3, max_source_mag=26):
        """
        Initialize GGSL criteria.
        
        Parameters
        ----------
        min_magnification : float
            Minimum total magnification to count as GGSL
        min_arc_length_arcsec : float
            Minimum arc length (arcsec)
        min_image_separation_arcsec : float
            Minimum separation between images (arcsec)
        max_source_mag : float
            Maximum source magnitude for detection
        """
        self.min_magnification = min_magnification
        self.min_arc_length_arcsec = min_arc_length_arcsec
        self.min_image_separation_arcsec = min_image_separation_arcsec
        self.max_source_mag = max_source_mag
    
    def is_ggsl_event(self, observable: GGSLObservable) -> bool:
        """
        Determine if an observable meets GGSL criteria.
        
        Parameters
        ----------
        observable : GGSLObservable
            Candidate GGSL event
        
        Returns
        -------
        is_ggsl : bool
            True if meets all criteria
        """
        # Check magnification
        if observable.total_magnification < self.min_magnification:
            return False
        
        # Check arc length
        if observable.arc_length_arcsec < self.min_arc_length_arcsec:
            return False
        
        # Check image separation (at least 2 images, separated)
        if observable.n_images < 2:
            return False
        
        # Check for strong arc morphology
        if not observable.is_strong_arc(min_length_arcsec=self.min_arc_length_arcsec,
                                        min_mag=self.min_magnification):
            return False
        
        return True
    
    def __repr__(self):
        return (f"GGSLDefinitions(mu_min={self.min_magnification}, "
                f"len_min={self.min_arc_length_arcsec}\")")


class GGSLCounter:
    """
    Count and characterize GGSL events in simulated images.
    """
    
    def __init__(self, deflection_field, magnification_calc, definitions: GGSLDefinitions):
        """
        Initialize GGSL counter.
        
        Parameters
        ----------
        deflection_field : DeflectionField
            Deflection field calculator
        magnification_calc : Magnification
            Magnification calculator
        definitions : GGSLDefinitions
            GGSL event criteria
        """
        self.deflection = deflection_field
        self.magnification = magnification_calc
        self.definitions = definitions
        self.events = []
    
    def find_images(self, source_position, x_grid, y_grid, tolerance=0.01):
        """
        Find all images of a source.
        
        Uses ray-tracing to find image positions.
        
        Parameters
        ----------
        source_position : tuple
            (x, y) source position in source plane
        x_grid, y_grid : ndarray
            Grid for ray tracing
        tolerance : float
            Convergence tolerance for image finding
        
        Returns
        -------
        images : list of tuple
            List of (x, y) image positions
        magnifications : list of float
            Magnifications of each image
        """
        images = []
        magnifications = []
        
        # Sample many points on the image plane
        X, Y = np.meshgrid(x_grid, y_grid)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        source_x, source_y = source_position
        
        # For each image plane point, compute its source plane position
        alpha_x, alpha_y = self.deflection.deflection_angle(X_flat, Y_flat)
        beta_x = X_flat - alpha_x
        beta_y = Y_flat - alpha_y
        
        # Find points near the source position
        distance = np.sqrt((beta_x - source_x)**2 + (beta_y - source_y)**2)
        near_source = distance < tolerance
        
        if np.any(near_source):
            image_x = X_flat[near_source]
            image_y = Y_flat[near_source]
            
            # Cluster nearby points (multiple detections of same image)
            unique_images = self._cluster_images(image_x, image_y, tolerance * 3)
            
            for img_x, img_y in unique_images:
                images.append((img_x, img_y))
                mag = self.magnification.magnification(img_x, img_y)
                magnifications.append(mag)
        
        return images, magnifications
    
    @staticmethod
    def _cluster_images(x, y, distance_threshold):
        """
        Cluster nearby image points.
        
        Parameters
        ----------
        x, y : ndarray
            Image coordinates
        distance_threshold : float
            Minimum distance to separate clusters
        
        Returns
        -------
        clusters : list of tuple
            List of (x_center, y_center) for each cluster
        """
        points = np.column_stack([x, y])
        clusters = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
            
            # Find all points near this one
            distances = np.linalg.norm(points - point, axis=1)
            cluster_mask = distances < distance_threshold
            cluster_indices = np.where(cluster_mask)[0]
            
            # Mark as used
            for idx in cluster_indices:
                used.add(idx)
            
            # Compute centroid
            centroid = np.mean(points[cluster_indices], axis=0)
            clusters.append(tuple(centroid))
        
        return clusters
    
    def characterize_images(self, images, magnifications, source_position):
        """
        Characterize the set of images (arc properties).
        
        Parameters
        ----------
        images : list of tuple
            Image positions
        magnifications : list of float
            Magnifications
        source_position : tuple
            Source position (for reference)
        
        Returns
        -------
        properties : dict
            Arc properties including length, width, etc.
        """
        if len(images) < 2:
            return None
        
        images = np.array(images)
        
        # Arc length: extent of images
        arc_length = np.max(np.ptp(images, axis=0))
        
        # Arc width: perpendicular extent (simplified)
        arc_width = np.min(np.ptp(images, axis=0))
        
        # Total magnification
        total_mag = np.sum(magnifications)
        
        return {
            'arc_length': arc_length,
            'arc_width': arc_width,
            'total_magnification': total_mag,
            'n_images': len(images),
            'image_positions': images,
            'magnifications': np.array(magnifications),
        }
    
    def detect_events(self, sources, x_grid, y_grid, source_z=1.0, lens_z=0.3):
        """
        Detect GGSL events in a source catalog.
        
        Parameters
        ----------
        sources : ndarray of shape (n_sources, 2)
            Source positions (x, y)
        x_grid, y_grid : ndarray
            Grid for ray tracing
        source_z : float
            Source redshift
        lens_z : float
            Lens redshift
        
        Returns
        -------
        events : list of GGSLObservable
            Detected GGSL events
        """
        self.events = []
        event_id = 0
        
        for source_id, source_pos in enumerate(sources):
            # Find images
            images, mags = self.find_images(source_pos, x_grid, y_grid)
            
            if len(images) < 2:
                continue
            
            # Characterize images
            props = self.characterize_images(images, mags, source_pos)
            
            if props is None:
                continue
            
            # Create observable
            observable = GGSLObservable(
                event_id=event_id,
                source_id=source_id,
                n_images=props['n_images'],
                magnifications=props['magnifications'],
                total_magnification=props['total_magnification'],
                arc_length_arcsec=props['arc_length'],
                arc_width_arcsec=props['arc_width'],
                source_redshift=source_z,
                lens_redshift=lens_z,
            )
            
            # Check if meets GGSL criteria
            if self.definitions.is_ggsl_event(observable):
                self.events.append(observable)
                event_id += 1
        
        return self.events
    
    def summary(self):
        """Get summary statistics of detected events."""
        if not self.events:
            return {
                'n_events': 0,
                'mean_magnification': 0,
                'mean_arc_length': 0,
            }
        
        mags = [e.total_magnification for e in self.events]
        lengths = [e.arc_length_arcsec for e in self.events]
        
        return {
            'n_events': len(self.events),
            'mean_magnification': np.mean(mags),
            'median_magnification': np.median(mags),
            'mean_arc_length': np.mean(lengths),
            'median_arc_length': np.median(lengths),
            'n_images_distribution': [e.n_images for e in self.events],
        }
