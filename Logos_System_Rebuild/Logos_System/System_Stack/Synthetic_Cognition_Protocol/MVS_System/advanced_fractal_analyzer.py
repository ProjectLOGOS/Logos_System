# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""
Fractal Orbit Analysis Toolkit - Advanced Mathematical Layer
===========================================================

This module provides advanced mathematical analysis layers for extracting
meaning from fractal iterations. It integrates multiple mathematical disciplines
to provide deep insights into complex system dynamics.

Analysis Layers:
- Topological Analysis: Shape, connectivity, and structural properties
- Information Theory: Complexity, entropy, and information content
- Dynamical Systems: Stability, periodicity, and chaotic behavior
- Statistical Analysis: Distribution patterns and fractal dimensions
- Spectral Analysis: Frequency domain characteristics
- Graph Theory: Network modeling of fractal boundaries
- Category Theory: Abstract relationships and morphisms
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import scipy.stats
import scipy.signal

@dataclass
class FractalAnalysisResult:
    """Container for comprehensive fractal analysis results"""
    topological: Dict[str, Any]
    information: Dict[str, float]
    dynamical: Dict[str, Any]
    statistical: Dict[str, Any]
    spectral: Dict[str, Any]
    graph: Dict[str, Any]
    categorical: Dict[str, Any]

class AdvancedFractalAnalyzer:
    """
    Advanced mathematical analyzer for fractal iteration data.
    Provides multi-layered analysis to extract meaning from complex patterns.
    """

    def __init__(self, iterations_data: np.ndarray):
        """
        Initialize analyzer with fractal iteration data.

        Args:
            iterations_data: 2D array of iteration counts from Julia/Mandelbrot computation
        """
        self.iterations = iterations_data
        self.height, self.width = iterations_data.shape
        self.max_iter = np.max(iterations_data)

        # Pre-compute useful masks and boundaries
        self.boundary_mask = self._extract_boundary()
        self.connected_components = self._find_connected_components()

    def analyze_all_layers(self) -> FractalAnalysisResult:
        """
        Perform comprehensive multi-layered analysis of the fractal.

        Returns:
            FractalAnalysisResult containing all analysis layers
        """
        return FractalAnalysisResult(
            topological=self._topological_analysis(),
            information=self._information_analysis(),
            dynamical=self._dynamical_analysis(),
            statistical=self._statistical_analysis(),
            spectral=self._spectral_analysis(),
            graph=self._graph_analysis(),
            categorical=self._categorical_analysis()
        )

    def _extract_boundary(self) -> np.ndarray:
        """Extract fractal boundary (points near the edge of convergence)"""
        # Points that escaped vs those that didn't
        escaped = self.iterations < self.max_iter
        # Boundary is where escaped and non-escaped regions meet
        kernel = np.ones((3, 3))
        boundary = scipy.signal.convolve2d(escaped.astype(int), kernel, mode='same')
        boundary = (boundary > 0) & (boundary < 9)  # Edge pixels
        return boundary

    def _find_connected_components(self) -> List[np.ndarray]:
        """Find connected components in the fractal structure"""
        from scipy import ndimage

        # Label connected components of escaped regions
        escaped = (self.iterations < self.max_iter).astype(int)
        labeled, num_components = ndimage.label(escaped)

        components = []
        for i in range(1, num_components + 1):
            component_mask = (labeled == i)
            components.append(component_mask)

        return components

    def _topological_analysis(self) -> Dict[str, Any]:
        """Topological analysis: holes, connectivity, genus"""
        analysis = {}

        # Betti numbers approximation (simplified)
        # H0: Connected components
        analysis['betti_0'] = len(self.connected_components)

        # H1: Holes (simplified: count enclosed regions)
        escaped_regions = (self.iterations < self.max_iter)
        total_escaped = np.sum(escaped_regions)

        # Estimate holes by looking for "islands" of non-escaped points
        # surrounded by escaped points
        holes = 0
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if not escaped_regions[i, j]:  # Non-escaped point
                    # Check if surrounded by escaped points
                    neighbors = escaped_regions[i-1:i+2, j-1:j+2]
                    if np.all(neighbors):  # All neighbors escaped
                        holes += 1

        analysis['betti_1'] = holes

        # Fractal dimension using box-counting method
        analysis['fractal_dimension'] = self._calculate_fractal_dimension()

        # Connectivity analysis
        analysis['connectivity'] = self._analyze_connectivity()

        return analysis

    def _calculate_fractal_dimension(self) -> float:
        """Calculate fractal dimension using box-counting method"""
        # Simplified box-counting for 2D data
        scales = [2, 4, 8, 16, 32]
        counts = []

        for scale in scales:
            # Count boxes that contain boundary points
            boxes = set()
            for i in range(0, self.height, scale):
                for j in range(0, self.width, scale):
                    if np.any(self.boundary_mask[i:i+scale, j:j+scale]):
                        boxes.add((i//scale, j//scale))
            counts.append(len(boxes))

        # Linear regression on log-log plot
        if len(counts) > 1:
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return -slope  # Fractal dimension

        return 2.0  # Default to 2D

    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze connectivity properties"""
        connectivity = {}

        # Component sizes
        component_sizes = [np.sum(comp) for comp in self.connected_components]
        connectivity['component_sizes'] = component_sizes
        connectivity['largest_component_ratio'] = max(component_sizes) / sum(component_sizes) if component_sizes else 0

        # Percolation analysis (simplified)
        connectivity['percolation'] = self._check_percolation()

        return connectivity

    def _check_percolation(self) -> bool:
        """Check if the fractal percolates (connects opposite sides)"""
        escaped = (self.iterations < self.max_iter)

        # Check horizontal percolation
        left_connected = np.any(escaped[:, 0])
        right_connected = np.any(escaped[:, -1])

        # Check vertical percolation
        top_connected = np.any(escaped[0, :])
        bottom_connected = np.any(escaped[-1, :])

        return (left_connected and right_connected) or (top_connected and bottom_connected)

    def _information_analysis(self) -> Dict[str, float]:
        """Information theory analysis: entropy, complexity"""
        analysis = {}

        # Shannon entropy of iteration distribution
        hist, _ = np.histogram(self.iterations.flatten(), bins=self.max_iter+1, density=True)
        hist = hist[hist > 0]  # Remove zeros
        analysis['shannon_entropy'] = -np.sum(hist * np.log2(hist))

        # Kolmogorov complexity approximation (compression ratio)
        flat_data = self.iterations.flatten()
        # Simple RLE compression as approximation
        compressed = self._simple_rle_compress(flat_data)
        analysis['compression_ratio'] = len(compressed) / len(flat_data)

        # Mutual information between neighboring pixels
        analysis['spatial_mutual_info'] = self._calculate_spatial_mutual_info()

        # Fractal information dimension
        analysis['information_dimension'] = self._calculate_information_dimension()

        return analysis

    def _simple_rle_compress(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """Simple run-length encoding for compression analysis"""
        compressed = []
        current_value = data[0]
        count = 1

        for value in data[1:]:
            if value == current_value:
                count += 1
            else:
                compressed.append((current_value, count))
                current_value = value
                count = 1

        compressed.append((current_value, count))
        return compressed

    def _calculate_spatial_mutual_info(self) -> float:
        """Calculate mutual information between neighboring pixels"""
        # Simplified: correlation between horizontal neighbors
        right_neighbors = self.iterations[:, 1:]
        left_pixels = self.iterations[:, :-1]

        # Joint histogram
        joint_hist, _, _ = np.histogram2d(left_pixels.flatten(), right_neighbors.flatten(),
                                        bins=[self.max_iter+1, self.max_iter+1])

        # Marginal histograms
        marginal1 = np.sum(joint_hist, axis=1)
        marginal2 = np.sum(joint_hist, axis=0)

        # Mutual information
        joint_hist = joint_hist / np.sum(joint_hist)
        marginal1 = marginal1 / np.sum(marginal1)
        marginal2 = marginal2 / np.sum(marginal2)

        mi = 0
        for i in range(len(marginal1)):
            for j in range(len(marginal2)):
                if joint_hist[i, j] > 0:
                    mi += joint_hist[i, j] * np.log2(joint_hist[i, j] / (marginal1[i] * marginal2[j]))

        return mi

    def _calculate_information_dimension(self) -> float:
        """Calculate information dimension"""
        # Use the fact that information dimension D_I = lim (I(r)/log(1/r)) as r->0
        # where I(r) is the information needed to specify the system at scale r

        scales = [2, 4, 8, 16]
        information_measures = []

        for scale in scales:
            # Divide into boxes and calculate information content
            boxes = []
            for i in range(0, self.height, scale):
                for j in range(0, self.width, scale):
                    box_data = self.iterations[i:i+scale, j:j+scale]
                    if box_data.size > 0:
                        # Information content of this box
                        hist, _ = np.histogram(box_data.flatten(), bins=min(10, self.max_iter+1), density=True)
                        hist = hist[hist > 0]
                        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
                        boxes.append(entropy)

            avg_info = np.mean(boxes) if boxes else 0
            information_measures.append(avg_info)

        # Linear regression
        if len(information_measures) > 1:
            log_scales = np.log(1/np.array(scales))
            slope, _ = np.polyfit(log_scales, information_measures, 1)
            return slope

        return 2.0

    def _dynamical_analysis(self) -> Dict[str, Any]:
        """Dynamical systems analysis: stability, periodicity"""
        analysis = {}

        # Lyapunov exponent approximation (simplified)
        analysis['lyapunov_exponent'] = self._estimate_lyapunov_exponent()

        # Periodicity analysis
        analysis['periodicity'] = self._analyze_periodicity()

        # Bifurcation analysis (simplified)
        analysis['bifurcation_points'] = self._find_bifurcation_points()

        # Stability analysis
        analysis['stability_regions'] = self._analyze_stability_regions()

        return analysis

    def _estimate_lyapunov_exponent(self) -> float:
        """Estimate Lyapunov exponent from iteration patterns"""
        # Simplified: use variance of iteration counts as proxy for chaos
        # Higher variance suggests more chaotic behavior
        variance = np.var(self.iterations.flatten())

        # Normalize and convert to Lyapunov-like measure
        # (This is a rough approximation)
        lyapunov = np.log(1 + variance / self.max_iter**2)
        return lyapunov

    def _analyze_periodicity(self) -> Dict[str, Any]:
        """Analyze periodic structures in the fractal"""
        periodicity = {}

        # Autocorrelation analysis
        flat_data = self.iterations.flatten()
        autocorr = np.correlate(flat_data, flat_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Second half

        # Find peaks in autocorrelation
        peaks = scipy.signal.find_peaks(autocorr, height=0.1)[0]
        periodicity['dominant_periods'] = peaks.tolist() if len(peaks) > 0 else []

        # Spectral analysis for periodicity
        fft = np.fft.fft(flat_data)
        freqs = np.fft.fftfreq(len(flat_data))
        power = np.abs(fft)**2

        # Find dominant frequencies
        peak_freqs = freqs[np.argsort(power)[-5:]]  # Top 5 frequencies
        periodicity['dominant_frequencies'] = peak_freqs.tolist()

        return periodicity

    def _find_bifurcation_points(self) -> List[Tuple[int, int]]:
        """Find points where iteration behavior changes dramatically"""
        # Look for high gradient regions in iteration counts
        grad_y, grad_x = np.gradient(self.iterations.astype(float))

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold for bifurcation points
        threshold = np.percentile(gradient_magnitude, 95)
        bifurcation_mask = gradient_magnitude > threshold

        # Return coordinates of bifurcation points
        bifurcation_points = np.where(bifurcation_mask)
        return list(zip(bifurcation_points[0], bifurcation_points[1]))

    def _analyze_stability_regions(self) -> Dict[str, Any]:
        """Analyze regions of different stability"""
        stability = {}

        # Classify regions by iteration count ranges
        low_iter = self.iterations < self.max_iter * 0.1
        mid_iter = (self.iterations >= self.max_iter * 0.1) & (self.iterations < self.max_iter * 0.5)
        high_iter = self.iterations >= self.max_iter * 0.5

        stability['stable_region_ratio'] = np.sum(low_iter) / self.iterations.size
        stability['chaotic_region_ratio'] = np.sum(high_iter) / self.iterations.size
        stability['transition_region_ratio'] = np.sum(mid_iter) / self.iterations.size

        return stability

    def _statistical_analysis(self) -> Dict[str, Any]:
        """Statistical analysis of iteration distributions"""
        analysis = {}

        flat_data = self.iterations.flatten()

        # Basic statistics
        analysis['mean_iterations'] = float(np.mean(flat_data))
        analysis['std_iterations'] = float(np.std(flat_data))
        analysis['median_iterations'] = float(np.median(flat_data))
        analysis['skewness'] = float(scipy.stats.skew(flat_data))
        analysis['kurtosis'] = float(scipy.stats.kurtosis(flat_data))

        # Distribution fitting
        analysis['distribution_fit'] = self._fit_distributions(flat_data)

        # Spatial statistics
        analysis['spatial_correlation'] = self._calculate_spatial_correlation()

        return analysis

    def _fit_distributions(self, data: np.ndarray) -> Dict[str, float]:
        """Fit various distributions to the iteration data"""
        fits = {}

        # Try different distributions
        distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm']

        for dist_name in distributions:
            try:
                dist = getattr(scipy.stats, dist_name)
                params = dist.fit(data)
                # Calculate AIC or similar goodness of fit
                log_likelihood = np.sum(dist.logpdf(data, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                fits[dist_name] = float(aic)
            except:
                fits[dist_name] = float('inf')

        return fits

    def _calculate_spatial_correlation(self) -> Dict[str, float]:
        """Calculate spatial correlation statistics (simplified for memory efficiency)"""
        correlation = {}

        # Simplified: calculate correlation between horizontal neighbors only
        flat_data = self.iterations.flatten()
        n = len(flat_data)

        # Horizontal neighbors correlation
        right_neighbors = self.iterations[:, 1:].flatten()
        left_pixels = self.iterations[:, :-1].flatten()

        if len(right_neighbors) > 0:
            horz_corr = np.corrcoef(left_pixels, right_neighbors)[0, 1]
        else:
            horz_corr = 0.0

        # Vertical neighbors correlation
        bottom_neighbors = self.iterations[1:, :].flatten()
        top_pixels = self.iterations[:-1, :].flatten()

        if len(bottom_neighbors) > 0:
            vert_corr = np.corrcoef(top_pixels, bottom_neighbors)[0, 1]
        else:
            vert_corr = 0.0

        correlation['horizontal_correlation'] = float(horz_corr)
        correlation['vertical_correlation'] = float(vert_corr)
        correlation['mean_spatial_correlation'] = float((horz_corr + vert_corr) / 2)

        return correlation

    def _spectral_analysis(self) -> Dict[str, Any]:
        """Spectral analysis: frequency domain characteristics"""
        analysis = {}

        # 2D Fourier transform
        fft2d = np.fft.fft2(self.iterations.astype(float))
        power_spectrum = np.abs(fft2d)**2

        # Shift zero frequency to center
        power_spectrum = np.fft.fftshift(power_spectrum)

        analysis['power_spectrum'] = power_spectrum.tolist()

        # Dominant frequencies
        max_power_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
        analysis['dominant_frequency_location'] = [int(max_power_idx[0]), int(max_power_idx[1])]

        # Spectral entropy
        normalized_power = power_spectrum / np.sum(power_spectrum)
        normalized_power = normalized_power[normalized_power > 0]
        spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power))
        analysis['spectral_entropy'] = float(spectral_entropy)

        # Fractal spectrum analysis (simplified)
        analysis['fractal_spectrum'] = self._calculate_fractal_spectrum()

        return analysis

    def _calculate_fractal_spectrum(self) -> Dict[str, Any]:
        """Calculate fractal spectrum (f(α) spectrum)"""
        spectrum = {}

        # Simplified multifractal spectrum calculation
        # This is a complex analysis - here we provide basic structure

        # Partition data into boxes and analyze local scaling
        scales = [2, 4, 8, 16]
        scaling_exponents = []

        for scale in scales:
            local_exponents = []
            for i in range(0, self.height, scale):
                for j in range(0, self.width, scale):
                    box_data = self.iterations[i:i+scale, j:j+scale]
                    if box_data.size > 0:
                        # Local scaling exponent (simplified)
                        mean_val = np.mean(box_data)
                        if mean_val > 0:
                            exponent = np.log(mean_val) / np.log(scale)
                            local_exponents.append(exponent)

            if local_exponents:
                scaling_exponents.append(np.mean(local_exponents))

        spectrum['scaling_exponents'] = scaling_exponents
        spectrum['multifractal_dimension'] = np.std(scaling_exponents) if scaling_exponents else 0

        return spectrum

    def _graph_analysis(self) -> Dict[str, Any]:
        """Graph theory analysis: network modeling of fractal structure"""
        analysis = {}

        # Model fractal boundary as a graph
        boundary_pixels = np.where(self.boundary_mask)
        boundary_coords = list(zip(boundary_pixels[0], boundary_pixels[1]))

        # Create graph where pixels are nodes, edges connect neighbors
        from scipy.spatial import cKDTree

        if len(boundary_coords) > 0:
            coords_array = np.array(boundary_coords)
            tree = cKDTree(coords_array)

            # Connect pixels within distance 1 (4-connectivity) or sqrt(2) (8-connectivity)
            distances, indices = tree.query(coords_array, k=5, distance_upper_bound=np.sqrt(2)+0.1)

            # Build adjacency list
            adjacency = defaultdict(list)
            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                for j, dist in zip(idx_row, dist_row):
                    if dist > 0 and dist <= np.sqrt(2)+0.1 and j < len(boundary_coords):
                        adjacency[i].append(j)

            analysis['num_nodes'] = len(boundary_coords)
            analysis['num_edges'] = sum(len(neighbors) for neighbors in adjacency.values()) // 2

            # Graph metrics
            degrees = [len(neighbors) for neighbors in adjacency.values()]
            analysis['average_degree'] = float(np.mean(degrees)) if degrees else 0
            analysis['max_degree'] = int(np.max(degrees)) if degrees else 0

            # Clustering coefficient approximation
            triangles = 0
            for node, neighbors in adjacency.items():
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        if neighbors[j] in adjacency[neighbors[i]]:
                            triangles += 1

            analysis['clustering_coefficient'] = triangles / len(boundary_coords) if boundary_coords else 0

        else:
            analysis['num_nodes'] = 0
            analysis['num_edges'] = 0
            analysis['average_degree'] = 0
            analysis['max_degree'] = 0
            analysis['clustering_coefficient'] = 0

        return analysis

    def _categorical_analysis(self) -> Dict[str, Any]:
        """Category theory analysis: abstract relationships and morphisms"""
        analysis = {}

        # Model fractal as a category
        # Objects: connected components
        # Morphisms: relationships between components

        if len(self.connected_components) > 0:
            # Category structure
            analysis['num_objects'] = len(self.connected_components)

            # Morphisms based on proximity and similarity
            morphisms = []
            component_centers = []

            for comp in self.connected_components:
                coords = np.where(comp)
                center = (np.mean(coords[0]), np.mean(coords[1]))
                component_centers.append(center)

            # Define morphisms between nearby components
            for i, center1 in enumerate(component_centers):
                for j, center2 in enumerate(component_centers):
                    if i != j:
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        if distance < min(self.height, self.width) / 4:  # Close components
                            morphisms.append((i, j, distance))

            analysis['num_morphisms'] = len(morphisms)

            # Functor analysis (simplified)
            # Check if structure preserves relationships
            analysis['structure_preservation'] = self._analyze_structure_preservation()

        else:
            analysis['num_objects'] = 0
            analysis['num_morphisms'] = 0
            analysis['structure_preservation'] = 0

        return analysis

    def _analyze_structure_preservation(self) -> float:
        """Analyze how well the fractal structure preserves mathematical relationships"""
        # Simplified: measure how well boundary structure matches expected fractal patterns

        # Calculate how "fractal-like" the boundary is
        boundary_length = np.sum(self.boundary_mask)

        # Expected boundary length for a fractal should scale with resolution
        # For a true fractal, boundary length increases with magnification

        # This is a rough measure - in practice would need multi-scale analysis
        perimeter_ratio = boundary_length / (2 * (self.height + self.width))

        # Values closer to 1 suggest more regular structures
        # Values > 1 suggest fractal boundaries
        return float(perimeter_ratio)


def analyze_fractal_iterations(iterations: np.ndarray) -> FractalAnalysisResult:
    """
    Convenience function to analyze fractal iteration data.

    Args:
        iterations: 2D numpy array of iteration counts

    Returns:
        Comprehensive analysis results
    """
    analyzer = AdvancedFractalAnalyzer(iterations)
    return analyzer.analyze_all_layers()


# Example usage and testing
if __name__ == '__main__':
    # Generate a simple Julia set for testing
    import matplotlib
    matplotlib.use('Agg')

    def generate_test_julia(c_value=complex(-0.7, 0.27015), width=200, height=150, max_iter=100):
        x = np.linspace(-2, 2, width)
        y = np.linspace(-1.5, 1.5, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        iterations = np.zeros(Z.shape, dtype=int)
        mask = np.ones(Z.shape, dtype=bool)

        for i in range(max_iter):
            Z[mask] = Z[mask]**2 + c_value
            escaped = np.abs(Z) > 2.0
            iterations[mask & escaped] = i
            mask &= ~escaped

        return iterations

    # Test the analyzer
    print("Testing Advanced Fractal Analyzer...")
    test_iterations = generate_test_julia()

    analyzer = AdvancedFractalAnalyzer(test_iterations)
    results = analyzer.analyze_all_layers()

    print("\nAnalysis Results Summary:")
    print(f"Topological: {len(results.topological)} metrics")
    print(f"Information: Shannon entropy = {results.information['shannon_entropy']:.3f}")
    print(f"Dynamical: Lyapunov exponent = {results.dynamical['lyapunov_exponent']:.3f}")
    print(f"Statistical: Mean iterations = {results.statistical['mean_iterations']:.1f}")
    print(f"Graph: {results.graph['num_nodes']} nodes, {results.graph['num_edges']} edges")
    print(f"Categorical: {results.categorical['num_objects']} objects")

    print("\nAdvanced fractal analysis completed successfully!")