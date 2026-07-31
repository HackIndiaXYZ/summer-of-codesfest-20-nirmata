"""
MRI QC Pipeline v3 — No Skull Stripping
Otsu masks + KMeans segmentation + 22 MRIQC IQMs + Artifact Detection
"""

import os
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage
from sklearn.cluster import KMeans

FEATURE_COLUMNS = [
    'cjv', 'cnr', 'efc', 'fber', 'fwhm_avg', 'inu_med', 'inu_range',
    'qi_1', 'qi_2', 'snr_csf', 'snr_gm', 'snr_total', 'snr_wm', 'snrd_total',
    'icvs_csf', 'icvs_gm', 'icvs_wm', 'wm2max',
    'summary_bg_mean', 'summary_bg_stdv', 'summary_gm_mean', 'summary_wm_mean'
]


# ==========================================================================
# Mask Estimation
# ==========================================================================

def otsu_threshold(data):
    nonzero = data[data > 0].flatten()
    if len(nonzero) == 0:
        return 0.0
    n_bins = 256
    hist, bin_edges = np.histogram(nonzero, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    sum_total = np.sum(bin_centers * hist)
    sum_bg = 0.0
    weight_bg = 0.0
    best_threshold = 0.0
    best_variance = 0.0
    for i in range(n_bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[i]
    return best_threshold


def compute_masks(data):
    threshold = otsu_threshold(data)
    head_mask = (data > threshold).astype(np.uint8)
    head_mask = ndimage.binary_fill_holes(head_mask).astype(np.uint8)
    labeled, n_components = ndimage.label(head_mask)
    if n_components > 1:
        component_sizes = ndimage.sum(head_mask, labeled, range(1, n_components + 1))
        largest_component = np.argmax(component_sizes) + 1
        head_mask = (labeled == largest_component).astype(np.uint8)

    air_mask = (data > 0) & (data <= threshold) & (head_mask == 0)
    if np.sum(air_mask) < 1000:
        air_nonzero = (head_mask == 0) & (data > 0)
        if np.sum(air_nonzero) > 500:
            air_mask = air_nonzero
        else:
            air_mask = (head_mask == 0) & (data >= 0)

    head_voxels = data[head_mask > 0]
    if len(head_voxels) > 0:
        brain_threshold = otsu_threshold(data * head_mask)
        brain_mask = ((data > brain_threshold) & (head_mask > 0)).astype(np.uint8)
        struct = ndimage.generate_binary_structure(3, 1)
        brain_mask = ndimage.binary_erosion(brain_mask, struct, iterations=2).astype(np.uint8)
        brain_mask = ndimage.binary_dilation(brain_mask, struct, iterations=2).astype(np.uint8)
        brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.uint8)
        labeled_brain, n_brain = ndimage.label(brain_mask)
        if n_brain > 1:
            sizes = ndimage.sum(brain_mask, labeled_brain, range(1, n_brain + 1))
            largest = np.argmax(sizes) + 1
            brain_mask = (labeled_brain == largest).astype(np.uint8)
    else:
        brain_mask = head_mask.copy()

    return air_mask.astype(bool), head_mask.astype(bool), brain_mask.astype(bool)


def detect_contrast(data, brain_mask):
    brain_voxels = data[brain_mask > 0]
    if len(brain_voxels) < 100:
        return 'T1w'
    sample_size = min(50000, len(brain_voxels))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(brain_voxels), sample_size, replace=False)
    sample = brain_voxels[sample_idx].reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(sample)
    centroids = kmeans.cluster_centers_.flatten()
    brightest_cluster = np.argmax(centroids)
    all_labels = kmeans.predict(brain_voxels.reshape(-1, 1))
    coords = np.array(np.where(brain_mask > 0)).T
    bright_coords = coords[all_labels == brightest_cluster]
    if len(bright_coords) == 0:
        return 'T1w'
    brain_center = coords.mean(axis=0)
    brain_dists = np.sqrt(np.sum((coords - brain_center) ** 2, axis=1))
    bright_dists = np.sqrt(np.sum((bright_coords - brain_center) ** 2, axis=1))
    ratio = np.mean(bright_dists) / (np.mean(brain_dists) + 1e-10)
    return 'T2w' if ratio > 1.05 else 'T1w'


# ==========================================================================
# Tissue Segmentation
# ==========================================================================

def segment_tissues(data, brain_mask, contrast='T1w'):
    brain_voxels = data[brain_mask > 0].reshape(-1, 1)
    if len(brain_voxels) < 100:
        return None
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(brain_voxels)
    centroids = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centroids)

    if contrast == 'T1w':
        csf_label = sorted_indices[0]
        gm_label = sorted_indices[1]
        wm_label = sorted_indices[2]
    else:
        wm_label = sorted_indices[0]
        gm_label = sorted_indices[1]
        csf_label = sorted_indices[2]

    full_labels = np.zeros(data.shape, dtype=np.int32)
    full_labels[brain_mask > 0] = labels + 1

    return {
        'wm_mask': (full_labels == wm_label + 1),
        'gm_mask': (full_labels == gm_label + 1),
        'csf_mask': (full_labels == csf_label + 1),
        'wm_mean': float(centroids[wm_label]),
        'gm_mean': float(centroids[gm_label]),
        'csf_mean': float(centroids[csf_label]),
        'wm_voxels': int(np.sum(full_labels == wm_label + 1)),
        'gm_voxels': int(np.sum(full_labels == gm_label + 1)),
        'csf_voxels': int(np.sum(full_labels == csf_label + 1)),
    }


# ==========================================================================
# 22 IQM Feature Extractors
# ==========================================================================

def compute_snr_tissue(data, tissue_mask, air_mask):
    tissue_voxels = data[tissue_mask]
    air_voxels = data[air_mask]
    if len(tissue_voxels) == 0 or len(air_voxels) < 10:
        return 0.0
    noise_std = np.std(air_voxels)
    if noise_std < 1e-10:
        return 0.0
    return float(np.mean(tissue_voxels) / noise_std)


def compute_snr_dietrich(data, brain_mask, air_mask):
    DIETRICH_FACTOR = 0.6551364
    foreground = data[brain_mask]
    air_voxels = data[air_mask]
    if len(foreground) == 0 or len(air_voxels) < 10:
        return 0.0
    mu_fg = np.mean(foreground)
    mad = np.median(np.abs(air_voxels - np.median(air_voxels)))
    sigma = mad * 1.4826
    if sigma < 1e-10:
        sigma = np.std(air_voxels)
    if sigma < 1e-10:
        return 0.0
    return float(DIETRICH_FACTOR * mu_fg / sigma)


def compute_cnr(data, wm_mask, gm_mask, air_mask):
    wm_voxels = data[wm_mask]
    gm_voxels = data[gm_mask]
    air_voxels = data[air_mask]
    if len(wm_voxels) == 0 or len(gm_voxels) == 0 or len(air_voxels) < 10:
        return 0.0
    noise_std = np.std(air_voxels)
    if noise_std < 1e-10:
        return 0.0
    return float(abs(np.mean(wm_voxels) - np.mean(gm_voxels)) / noise_std)


def compute_cjv(data, wm_mask, gm_mask):
    wm_voxels = data[wm_mask]
    gm_voxels = data[gm_mask]
    if len(wm_voxels) == 0 or len(gm_voxels) == 0:
        return 1.0
    contrast = abs(np.mean(wm_voxels) - np.mean(gm_voxels))
    if contrast < 1e-10:
        return 10.0
    return float((np.std(wm_voxels) + np.std(gm_voxels)) / contrast)


def compute_efc(data, brain_mask):
    brain_data = np.abs(data[brain_mask > 0])
    if len(brain_data) == 0:
        return 1.0
    b_max = np.sqrt(np.sum(brain_data ** 2))
    if b_max < 1e-10:
        return 1.0
    data_norm = brain_data / b_max
    efc = -float(np.sum(data_norm * np.log(data_norm + 1e-16)))
    n = len(brain_data)
    efc_max = np.sqrt(n) * (0.5 * np.log(n))
    if efc_max < 1e-10:
        return 1.0
    return float(np.clip(efc / efc_max, 0.0, 1.0))


def compute_fber(data, brain_mask, air_mask):
    fg = data[brain_mask]
    air = data[air_mask]
    if len(fg) == 0 or len(air) < 10:
        return 0.0
    fg_energy = float(np.mean(fg ** 2))
    air_energy = float(np.mean(air ** 2))
    if air_energy < 1e-10:
        return 0.0
    return float(fg_energy / air_energy)


def compute_fwhm(data, brain_mask, voxel_sizes):
    masked = data * brain_mask
    fwhm_dims = []
    for axis in range(3):
        diff = np.diff(masked, axis=axis)
        var_diff = np.var(diff[diff != 0]) if np.any(diff != 0) else 1.0
        var_signal = np.var(masked[masked != 0]) if np.any(masked != 0) else 1.0
        if var_diff < 1e-10 or var_signal < 1e-10:
            fwhm_dims.append(0.0)
            continue
        ratio = var_diff / (2.0 * var_signal)
        if ratio >= 1.0:
            fwhm_dims.append(0.0)
            continue
        sigma_sq = -1.0 / (4.0 * np.log(1.0 - ratio + 1e-10))
        sigma = np.sqrt(max(0, sigma_sq))
        fwhm_dims.append(float(sigma * 2.355 * voxel_sizes[axis]))
    return float(np.mean(fwhm_dims))


def compute_qi1(data, air_mask):
    air_voxels = data[air_mask]
    air_positive = air_voxels[air_voxels > 0]
    if len(air_positive) < 10:
        return 0.0
    mode_est = np.sqrt(2.0 / np.pi) * np.mean(air_positive)
    threshold = 2.0 * mode_est
    artifact_count = np.sum(air_positive > threshold)
    return float(artifact_count) / float(len(air_positive))


def compute_qi2(data, air_mask):
    air_voxels = data[air_mask]
    air_positive = air_voxels[air_voxels > 0]
    if len(air_positive) < 100:
        return 0.0
    sorted_air = np.sort(air_positive)
    n = len(sorted_air)
    empirical_cdf = np.arange(1, n + 1) / n
    sigma = np.sqrt(np.mean(air_positive ** 2) / 2.0)
    theoretical_cdf = 1.0 - np.exp(-(sorted_air ** 2) / (2.0 * sigma ** 2))
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def compute_inu(data, brain_mask):
    masked_brain = data[brain_mask > 0]
    if len(masked_brain) == 0:
        return 1.0, 0.0
    global_median = np.median(masked_brain)
    if global_median < 1e-10:
        return 1.0, 0.0
    shape = data.shape
    block_size = max(10, min(shape) // 5)
    local_ratios = []
    for x in range(0, shape[0], block_size):
        for y in range(0, shape[1], block_size):
            for z in range(0, shape[2], block_size):
                block = data[x:x+block_size, y:y+block_size, z:z+block_size]
                block_mask = brain_mask[x:x+block_size, y:y+block_size, z:z+block_size]
                block_voxels = block[block_mask > 0]
                if len(block_voxels) > 20:
                    local_ratios.append(np.median(block_voxels) / global_median)
    if not local_ratios:
        return 1.0, 0.0
    ratios = np.array(local_ratios)
    return float(np.median(ratios)), float(np.percentile(ratios, 95) - np.percentile(ratios, 5))


def compute_icvs(segmentation):
    total = segmentation['wm_voxels'] + segmentation['gm_voxels'] + segmentation['csf_voxels']
    if total == 0:
        return 0.0, 0.0, 0.0
    return (float(segmentation['wm_voxels']) / total,
            float(segmentation['gm_voxels']) / total,
            float(segmentation['csf_voxels']) / total)


def compute_wm2max(data, wm_mask, brain_mask):
    wm_voxels = data[wm_mask]
    brain_voxels = data[brain_mask]
    if len(wm_voxels) == 0 or len(brain_voxels) == 0:
        return 0.0
    p995 = np.percentile(brain_voxels, 99.5)
    if p995 < 1e-10:
        return 0.0
    return float(np.median(wm_voxels) / p995)


def compute_summary_stats(data, air_mask, segmentation):
    air = data[air_mask]
    gm = data[segmentation['gm_mask']]
    wm = data[segmentation['wm_mask']]
    return {
        'summary_bg_mean': float(np.mean(air)) if len(air) > 0 else 0.0,
        'summary_bg_stdv': float(np.std(air)) if len(air) > 0 else 0.0,
        'summary_gm_mean': float(np.mean(gm)) if len(gm) > 0 else 0.0,
        'summary_wm_mean': float(np.mean(wm)) if len(wm) > 0 else 0.0,
    }


# ==========================================================================
# Artifact Detection
# ==========================================================================

def detect_motion_artifact(data, head_mask):
    brain_data = data * head_mask
    slices = []
    mid_ax = data.shape[2] // 2
    mid_cor = data.shape[1] // 2
    mid_sag = data.shape[0] // 2

    for i in range(-3, 4):
        z = mid_ax + i
        if 0 <= z < data.shape[2]:
            slices.append(brain_data[:, :, z])
        y = mid_cor + i
        if 0 <= y < data.shape[1]:
            slices.append(brain_data[:, y, :])
        x = mid_sag + i
        if 0 <= x < data.shape[0]:
            slices.append(brain_data[x, :, :])

    gradient_vars = []
    for s in slices:
        smax = np.max(s)
        if smax == 0:
            continue
        s_norm = s * (1.0 / smax)
        gy, gx = np.gradient(s_norm)
        gradient_vars.append(np.var(gx * gx + gy * gy))

    if not gradient_vars:
        return "none", 0.0

    gv = np.array(gradient_vars)
    inter_var = float(np.var(gv)) / (float(np.mean(gv)) + 1e-10)

    total_voxels = np.sum(head_mask)
    neg_fraction = np.sum(data < -5) / (total_voxels + 1e-10)

    score = 0.0
    if neg_fraction > 0.03: score += 3.0
    elif neg_fraction > 0.01: score += 2.0
    elif neg_fraction > 0.002: score += 1.0
    if inter_var > 0.25: score += 2.0
    elif inter_var > 0.15: score += 1.0
    elif inter_var > 0.08: score += 0.5

    if score >= 3.0: return "severe", score
    elif score >= 2.0: return "moderate", score
    elif score >= 1.0: return "mild", score
    return "none", score


def detect_ringing_artifact(data, head_mask):
    mid_slice = (data * head_mask)[:, :, data.shape[2] // 2]
    smax = np.max(mid_slice)
    if smax == 0:
        return "none", 0.0

    mag = np.abs(np.fft.fftshift(np.fft.fft2(mid_slice * (1.0 / smax))))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 3

    y, x = np.ogrid[:h, :w]
    outer = (x - cx)**2 + (y - cy)**2 > radius * radius

    hf_ratio = float(np.sum(mag[outer])) / (float(np.sum(mag)) + 1e-10)

    if hf_ratio > 0.55: return "severe", hf_ratio
    elif hf_ratio > 0.45: return "moderate", hf_ratio
    elif hf_ratio > 0.38: return "mild", hf_ratio
    return "none", hf_ratio


def detect_noise_artifact(snr_total):
    if snr_total < 5: return "severe", snr_total
    elif snr_total < 8: return "moderate", snr_total
    elif snr_total < 12: return "mild", snr_total
    return "none", snr_total


def detect_blur_artifact(fwhm_avg, efc):
    score = 0.0
    if fwhm_avg > 6.0: score += 3.0
    elif fwhm_avg > 4.5: score += 2.0
    elif fwhm_avg > 3.5: score += 1.0

    if efc > 0.65: score += 2.0
    elif efc > 0.58: score += 1.0
    elif efc > 0.52: score += 0.5

    combined = (fwhm_avg + efc * 10) / 2.0

    if score >= 4.0: return "severe", combined
    elif score >= 2.5: return "moderate", combined
    elif score >= 1.0: return "mild", combined
    return "none", combined


# ==========================================================================
# Main extraction function
# ==========================================================================

def extract_scan(filepath):
    """Full pipeline: load NIfTI -> masks -> segment -> 22 IQMs -> artifacts.
    Returns dict with all results or None on failure."""
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        voxel_sizes = img.header.get_zooms()[:3]

        air_mask, head_mask, brain_mask = compute_masks(data)
        contrast = detect_contrast(data, brain_mask)
        seg = segment_tissues(data, brain_mask, contrast=contrast)
        if seg is None:
            return None

        features = {}
        features['snr_total'] = compute_snr_tissue(data, brain_mask, air_mask)
        features['snr_wm'] = compute_snr_tissue(data, seg['wm_mask'], air_mask)
        features['snr_gm'] = compute_snr_tissue(data, seg['gm_mask'], air_mask)
        features['snr_csf'] = compute_snr_tissue(data, seg['csf_mask'], air_mask)
        features['snrd_total'] = compute_snr_dietrich(data, brain_mask, air_mask)
        features['cnr'] = compute_cnr(data, seg['wm_mask'], seg['gm_mask'], air_mask)
        features['cjv'] = compute_cjv(data, seg['wm_mask'], seg['gm_mask'])
        features['wm2max'] = compute_wm2max(data, seg['wm_mask'], brain_mask)
        features['efc'] = compute_efc(data, brain_mask)
        features['fwhm_avg'] = compute_fwhm(data, brain_mask, voxel_sizes)
        features['fber'] = compute_fber(data, brain_mask, air_mask)
        features['qi_1'] = compute_qi1(data, air_mask)
        features['qi_2'] = compute_qi2(data, air_mask)
        inu_med, inu_range = compute_inu(data, brain_mask)
        features['inu_med'] = inu_med
        features['inu_range'] = inu_range
        icvs_wm, icvs_gm, icvs_csf = compute_icvs(seg)
        features['icvs_wm'] = icvs_wm
        features['icvs_gm'] = icvs_gm
        features['icvs_csf'] = icvs_csf
        summary = compute_summary_stats(data, air_mask, seg)
        features.update(summary)

        # Artifacts
        m_sev, m_val = detect_motion_artifact(data, head_mask)
        r_sev, r_val = detect_ringing_artifact(data, head_mask)
        n_sev, n_val = detect_noise_artifact(features['snr_total'])
        b_sev, b_val = detect_blur_artifact(features['fwhm_avg'], features['efc'])

        artifacts = {
            'motion': {'severity': m_sev, 'score': round(m_val, 4)},
            'ringing': {'severity': r_sev, 'score': round(r_val, 4)},
            'noise': {'severity': n_sev, 'score': round(n_val, 4)},
            'blur': {'severity': b_sev, 'score': round(b_val, 4)},
        }

        meta = {
            'contrast': contrast,
            'n_brain_voxels': int(np.sum(brain_mask)),
            'brain_volume_cm3': round(float(np.sum(brain_mask)) * float(np.prod(voxel_sizes)) / 1000.0, 1),
            'shape': data.shape,
        }

        return {
            'features': features,
            'artifacts': artifacts,
            'meta': meta,
            'data': data,
            'head_mask': head_mask,
            'brain_mask': brain_mask,
        }

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def find_nifti_files(directory):
    """Find all .nii and .nii.gz files recursively."""
    files = sorted(
        glob.glob(os.path.join(directory, "**/*.nii"), recursive=True) +
        glob.glob(os.path.join(directory, "**/*.nii.gz"), recursive=True)
    )
    return [f for f in files if os.path.isfile(f)]
