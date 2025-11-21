"""
Cole-Cole Feature Extraction from Impedance Spectroscopy Data

This script processes CSV files containing impedance measurements and computes
Cole-Cole parameters (R0, R∞, τ, α) for each electrode configuration, separately
for each side (left/right).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import os
import glob
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def complex_impedance_from_polar(magnitude: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """
    Convert impedance from polar form (magnitude, phase) to complex form.
    
    Args:
        magnitude: Impedance magnitude (|Z|)
        phase_deg: Phase angle in degrees
    
    Returns:
        Complex impedance array
    """
    phase_rad = np.deg2rad(phase_deg)
    return magnitude * np.exp(1j * phase_rad)


def cole_cole_model(freq: np.ndarray, R0: float, Rinf: float, tau: float, alpha: float) -> np.ndarray:
    """
    Cole-Cole impedance model: Z(ω) = R∞ + (R0 - R∞) / (1 + (jωτ)^α)
    
    Args:
        freq: Frequency array (Hz)
        R0: Resistance at zero frequency (Ω)
        Rinf: Resistance at infinite frequency (Ω)
        tau: Relaxation time constant (s)
        alpha: Distribution parameter (0 < α ≤ 1)
    
    Returns:
        Complex impedance array
    """
    omega = 2 * np.pi * freq
    j_omega_tau = 1j * omega * tau
    j_omega_tau_alpha = np.power(j_omega_tau, alpha)
    Z = Rinf + (R0 - Rinf) / (1 + j_omega_tau_alpha)
    return Z


def cole_cole_residual(params: Tuple, freq: np.ndarray, Z_real: np.ndarray, Z_imag: np.ndarray) -> np.ndarray:
    """
    Residual function for Cole-Cole fitting.
    
    Args:
        params: (R0, Rinf, tau, alpha)
        freq: Frequency array
        Z_real: Real part of measured impedance
        Z_imag: Imaginary part of measured impedance
    
    Returns:
        Residual array (flattened real and imaginary parts)
    """
    R0, Rinf, tau, alpha = params
    Z_model = cole_cole_model(freq, R0, Rinf, tau, alpha)
    residual = np.concatenate([Z_real - Z_model.real, Z_imag - Z_model.imag])
    return residual


def fit_cole_cole(freq: np.ndarray, Z_complex: np.ndarray, debug: bool = False) -> Tuple[float, float, float, float, float, float, float]:
    """
    Fit Cole-Cole model to impedance data.
    
    Args:
        freq: Frequency array (Hz)
        Z_complex: Complex impedance array
        debug: If True, print diagnostic information when fit fails
    
    Returns:
        Tuple of (R0, Rinf, tau, alpha, rmse, relative_rmse, r_squared) or 
        (None, None, None, None, None, None, None) if fit fails
    """
    # Extract real and imaginary parts
    Z_real = Z_complex.real
    Z_imag = Z_complex.imag
    
    # Initial parameter estimates
    R0_est = np.max(Z_real)  # Maximum resistance (typically at low freq)
    Rinf_est = np.min(Z_real)  # Minimum resistance (typically at high freq)
    
    # Estimate tau from frequency at maximum -Z_imag
    if len(Z_imag) > 0:
        max_neg_imag_idx = np.argmax(-Z_imag)
        if max_neg_imag_idx < len(freq) and freq[max_neg_imag_idx] > 0:
            tau_est = 1.0 / (2 * np.pi * freq[max_neg_imag_idx])
        else:
            tau_est = 1e-4
    else:
        tau_est = 1e-4
    
    alpha_est = 0.7  # Typical value
    
    # Ensure R0 > Rinf
    if R0_est <= Rinf_est:
        R0_est = Rinf_est * 1.5
    
    # Bounds for parameters
    bounds = (
        (Rinf_est * 0.5, Rinf_est * 0.5, 1e-6, 0.1),  # Lower bounds
        (R0_est * 2, R0_est * 2, 1e-1, 1.0)  # Upper bounds
    )
    
    initial_params = [R0_est, Rinf_est, tau_est, alpha_est]
    
    try:
        # Fit using least squares
        popt, _ = curve_fit(
            lambda f, R0, Rinf, tau, alpha: np.concatenate([
                cole_cole_model(f, R0, Rinf, tau, alpha).real,
                cole_cole_model(f, R0, Rinf, tau, alpha).imag
            ]),
            freq,
            np.concatenate([Z_real, Z_imag]),
            p0=initial_params,
            bounds=bounds,
            maxfev=5000,
            method='trf'
        )
        
        R0, Rinf, tau, alpha = popt
        
        # Calculate RMSE
        Z_fitted = cole_cole_model(freq, R0, Rinf, tau, alpha)
        rmse = np.sqrt(np.mean(np.abs(Z_complex - Z_fitted)**2))
        
        # Calculate relative RMSE (as percentage)
        mean_impedance = np.mean(np.abs(Z_complex))
        relative_rmse = (rmse / mean_impedance) * 100 if mean_impedance > 0 else float('inf')
        
        # Calculate R-squared (coefficient of determination)
        ss_res = np.sum(np.abs(Z_complex - Z_fitted)**2)
        ss_tot = np.sum(np.abs(Z_complex - np.mean(Z_complex))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Validate parameters
        if R0 <= Rinf or alpha <= 0 or alpha > 1 or tau <= 0:
            if debug:
                print(f"      Fit failed validation: R0={R0:.1f}, Rinf={Rinf:.1f}, "
                      f"alpha={alpha:.3f}, tau={tau:.2e}")
                if R0 <= Rinf:
                    print(f"        → R0 ({R0:.1f}) <= Rinf ({Rinf:.1f})")
                if alpha <= 0 or alpha > 1:
                    print(f"        → alpha ({alpha:.3f}) not in (0, 1]")
                if tau <= 0:
                    print(f"        → tau ({tau:.2e}) <= 0")
            return None, None, None, None, None, None, None
        
        return R0, Rinf, tau, alpha, rmse, relative_rmse, r_squared
    
    except Exception as e:
        if debug:
            print(f"      Fit failed with exception: {type(e).__name__}: {str(e)}")
            print(f"        Data: {len(freq)} points, freq range: {np.min(freq):.0f}-{np.max(freq):.0f} Hz")
            print(f"        Z range: {np.min(np.abs(Z_complex)):.1f}-{np.max(np.abs(Z_complex)):.1f} Ω")
            print(f"        Initial estimates: R0={R0_est:.1f}, Rinf={Rinf_est:.1f}, tau={tau_est:.2e}, alpha={alpha_est:.2f}")
        return None, None, None, None, None, None, None


def assess_fit_quality(relative_rmse: float, r_squared: float, n_points: int) -> str:
    """
    Assess fit quality based on relative RMSE, R-squared, and number of data points.
    
    Args:
        relative_rmse: Relative RMSE as percentage
        r_squared: R-squared coefficient of determination
        n_points: Number of data points used in the fit
    
    Returns:
        Quality score: "good", "fair", or "poor"
    """
    # Check number of points
    if n_points < 5:
        return "poor"
    
    # Check relative error and R-squared
    if relative_rmse < 5 and r_squared > 0.95 and n_points >= 10:
        return "good"
    elif relative_rmse < 10 and r_squared > 0.90 and n_points >= 5:
        return "fair"
    else:
        return "poor"


def compute_cole_cole_features(df: pd.DataFrame, save_plots_dir: Optional[str] = None, 
                               patient_id: str = "", debug: bool = False) -> pd.DataFrame:
    """
    Compute Cole-Cole features for each electrode configuration, separately for each side.
    
    Args:
        df: DataFrame with columns: v1, v2, i1, i2, freq, z, phase, side
        save_plots_dir: Optional directory to save visualization plots
        patient_id: Patient ID for plot titles
        debug: If True, print diagnostic information about skipped configurations
    
    Returns:
        DataFrame with Cole-Cole parameters for each electrode configuration and side
    """
    results = []
    skipped_configs = []  # Track skipped configurations for debugging
    
    # Check if 'side' column exists, if not, create a dummy 'side' column
    if 'side' not in df.columns:
        df = df.copy()
        df['side'] = 'unknown'
    
    # Group by side and electrode configuration
    electrode_groups = df.groupby(['side', 'v1', 'v2', 'i1', 'i2'])
    
    for (side, v1, v2, i1, i2), group in electrode_groups:
        # Filter out disconnected measurements
        group_connected = group[group['is_disconnected'] == 0].copy()
        
        if len(group_connected) < 3:  # Need at least 3 points for fitting
            if debug:
                skipped_configs.append({
                    'side': side, 'v1': v1, 'v2': v2, 'i1': i1, 'i2': i2,
                    'reason': f'Insufficient points: {len(group_connected)} (need >= 3)',
                    'total_points': len(group),
                    'connected_points': len(group_connected)
                })
            continue
        
        # Sort by frequency
        group_connected = group_connected.sort_values('freq')
        
        # Extract data
        freq = group_connected['freq'].values
        z_magnitude = group_connected['z'].values
        phase_deg = group_connected['phase'].values
        
        # Convert to complex impedance
        Z_complex = complex_impedance_from_polar(z_magnitude, phase_deg)
        
        # Fit Cole-Cole model
        R0, Rinf, tau, alpha, rmse, relative_rmse, r_squared = fit_cole_cole(freq, Z_complex, debug=debug)
        
        if R0 is not None:
            # Calculate additional derived features
            delta_R = R0 - Rinf  # Resistance change
            fc = 1.0 / (2 * np.pi * tau) if tau > 0 else None  # Characteristic frequency
            
            # Assess fit quality
            n_points = len(group_connected)
            fit_quality = assess_fit_quality(relative_rmse, r_squared, n_points)
            
            results.append({
                'side': side,
                'v1': v1,
                'v2': v2,
                'i1': i1,
                'i2': i2,
                'R0': R0,
                'Rinf': Rinf,
                'tau': tau,
                'alpha': alpha,
                'delta_R': delta_R,
                'fc': fc,
                'rmse': rmse,
                'relative_rmse': relative_rmse,
                'r_squared': r_squared,
                'fit_quality': fit_quality,
                'n_points': n_points,
                'freq_min': np.min(freq),
                'freq_max': np.max(freq)
            })
        else:
            if debug:
                skipped_configs.append({
                    'side': side, 'v1': v1, 'v2': v2, 'i1': i1, 'i2': i2,
                    'reason': 'Cole-Cole fit failed',
                    'total_points': len(group),
                    'connected_points': len(group_connected)
                })
    
    # Print diagnostic information if debug mode
    if debug and skipped_configs:
        print(f"\n  Skipped {len(skipped_configs)} configurations:")
        for skip in skipped_configs[:20]:  # Show first 20
            print(f"    {skip['side']} side, v1={skip['v1']}, v2={skip['v2']}, "
                  f"i1={skip['i1']}, i2={skip['i2']}: {skip['reason']} "
                  f"(total: {skip['total_points']}, connected: {skip['connected_points']})")
        if len(skipped_configs) > 20:
            print(f"    ... and {len(skipped_configs) - 20} more")
    
    return pd.DataFrame(results)


def process_csv_file(filepath: str, save_plots_dir: Optional[str] = None, debug: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Process a single CSV file and compute Cole-Cole features separately for each side.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Tuple of (original_data, cole_cole_features)
        Features are computed separately for each side and electrode configuration.
    """
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Check required columns
        required_cols = ['v1', 'v2', 'i1', 'i2', 'freq', 'z', 'phase']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in {filepath}: {missing_cols}")
            return df, None
        
        # Get patient ID for plot titles
        patient_id = df['patient_id'].iloc[0] if 'patient_id' in df.columns and len(df) > 0 else ""
        
        # Compute Cole-Cole features (side is now included in the grouping)
        features_df = compute_cole_cole_features(df, save_plots_dir=save_plots_dir, 
                                             patient_id=patient_id, debug=debug)
        
        # Add metadata from original file
        if 'patient_id' in df.columns and len(df) > 0 and len(features_df) > 0:
            features_df['patient_id'] = df['patient_id'].iloc[0]
        
        return df, features_df
    
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None, None


def main(only_axilla: bool = True):
    """
    Main function to process all CSV files in patients_raw_data directory.
    
    Args:
        only_axilla: If True, filter to only axilla electrode configurations.
                     If False, use all configurations.
    """
    # Get all CSV files (focus on 'I' files which contain impedance data)
    data_dir = 'patients_raw_data'
    csv_files = glob.glob(os.path.join(data_dir, '*I.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} impedance CSV files")
    
    # Process each file
    all_features = []
    
    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        
        df, features_df = process_csv_file(filepath,debug=True)
        
        if features_df is not None and len(features_df) > 0:
            features_df['source_file'] = filename
            all_features.append(features_df)
            print(f"  Extracted {len(features_df)} electrode configurations")
        else:
            print(f"  No valid Cole-Cole fits found")
    
    # Combine all results
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Filter for specific electrode configurations for axilla (if requested)
        if only_axilla:
            target_configs = [
                (15, 23, 15, 23),  # v1,v2,i1,i2
            ]
            
            # Create a mask for filtering
            mask = pd.Series([False] * len(combined_features), index=combined_features.index)
            for v1,v2,i1,i2 in target_configs:
                mask |= ((combined_features['i1'] == i1) & 
                        (combined_features['v1'] == v1) &
                        (combined_features['i2'] == i2) &
                        (combined_features['v2'] == v2))
            
            combined_features = combined_features[mask].copy()
            
            print(f"\nFiltered to {len(combined_features)} configurations matching target electrode pairs:")
            for v1,v2,i1,i2 in target_configs:
                count = len(combined_features[
                    (combined_features['i1'] == i1) & 
                    (combined_features['v1'] == v1) &
                    (combined_features['i2'] == i2) &
                    (combined_features['v2'] == v2)
                ])
                print(f"  i1={i1}, v1={v1}, i2={i2}, v2={v2}: {count} fits")
        else:
            print(f"\nUsing all {len(combined_features)} configurations (only_axilla=False)")
        
        # Save results
        output_file = 'cole_cole_features.csv'
        combined_features.to_csv(output_file, index=False)
        print(f"\n✓ Saved Cole-Cole features to {output_file}")
        print(f"  Total configurations: {len(combined_features)}")
        print(f"\nFeature summary:")
        print(combined_features[['R0', 'Rinf', 'tau', 'alpha', 'delta_R', 'fc']].describe())
        
        # Add this after line 493, before the "="*60 line

        # Print detailed poor quality fits by patient
        poor_fits = combined_features[combined_features['fit_quality'] == 'poor'].copy()
        
        if len(poor_fits) > 0:
            print(f"\n{'='*60}")
            print("POOR QUALITY FITS BY PATIENT")
            print(f"{'='*60}")
            
            # Group by patient if patient_id column exists
            if 'patient_id' in poor_fits.columns:
                for patient_id in sorted(poor_fits['patient_id'].unique()):
                    patient_poor = poor_fits[poor_fits['patient_id'] == patient_id]
                    print(f"\n{patient_id} - {len(patient_poor)} poor quality fits:")
                    print(f"  {'Side':<6} {'v1':<3} {'v2':<3} {'i1':<3} {'i2':<3} "
                            f"{'Rel.RMSE':<10} {'R²':<8} {'n_points':<10} {'Reason':<15}")
                    print(f"  {'-'*6} {'-'*3} {'-'*3} {'-'*3} {'-'*3} "
                            f"{'-'*10} {'-'*8} {'-'*10} {'-'*15}")
                    
                    for idx, row in patient_poor.iterrows():
                        # Determine reason for poor fit
                        reasons = []
                        if row['n_points'] < 5:
                            reasons.append("Few points")
                        if row['relative_rmse'] > 10:
                            reasons.append("High error")
                        if row['r_squared'] < 0.90:
                            reasons.append("Low R²")
                        reason_str = ", ".join(reasons) if reasons else "Multiple"
                        
                        print(f"  {str(row['side']):<6} {int(row['v1']):<3} {int(row['v2']):<3} "
                                f"{int(row['i1']):<3} {int(row['i2']):<3} "
                                f"{row['relative_rmse']:>8.2f}% {row['r_squared']:>7.4f} "
                                f"{int(row['n_points']):>10} {reason_str:<15}")
            else:
                # If no patient_id, just print all poor fits
                print(f"\n{len(poor_fits)} poor quality fits (no patient_id column):")
                print(f"  {'Side':<6} {'v1':<3} {'v2':<3} {'i1':<3} {'i2':<3} "
                        f"{'Rel.RMSE':<10} {'R²':<8} {'n_points':<10} {'Reason':<15}")
                print(f"  {'-'*6} {'-'*3} {'-'*3} {'-'*3} {'-'*3} "
                        f"{'-'*10} {'-'*8} {'-'*10} {'-'*15}")
                
                for idx, row in poor_fits.iterrows():
                    reasons = []
                    if row['n_points'] < 5:
                        reasons.append("Few points")
                    if row['relative_rmse'] > 10:
                        reasons.append("High error")
                    if row['r_squared'] < 0.90:
                        reasons.append("Low R²")
                    reason_str = ", ".join(reasons) if reasons else "Multiple"
                    
                    print(f"  {str(row['side']):<6} {int(row['v1']):<3} {int(row['v2']):<3} "
                            f"{int(row['i1']):<3} {int(row['i2']):<3} "
                            f"{row['relative_rmse']:>8.2f}% {row['r_squared']:>7.4f} "
                            f"{int(row['n_points']):>10} {reason_str:<15}")
    
        print(f"\n{'='*60}")
        
        return combined_features
    else:
        print("\nNo features extracted from any files")
        return None


if __name__ == '__main__':
    features = main(only_axilla=False)