import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML
import tqdm

from slsim.ImageSimulation.image_simulation import simulate_image, rgb_image_from_image_list
from slsim.ImageSimulation.image_quality_lenstronomy import get_observatory

# =====================================================================
# 1. Data Generation & Processing
# =====================================================================
def generate_simulation_data(lens_class, time_array, bands, rgb_bands, num_pix, rgb_stretch, transform_matrix, cadence=None):
    """Simulates images, extracts lightcurves, and pre-computes all visual arrays."""
    print("\nGenerating simulation data...")
    num_images = lens_class.image_number[0]
    
    image_observation_times = lens_class.image_observer_times(time_array)
    time_steps = image_observation_times[0] 
    arrival_times = lens_class.point_source_arrival_times()[0]
    
    data = {
        'images': {b: [] for b in bands},
        'diff_images': {b: [] for b in bands},
        'rgb_images': [],
        'diff_rgb_images': [],
        'lightcurves': {b: [] for b in bands},
        'common_time': time_array,
        'image_times': image_observation_times,
        'positions': lens_class.point_source_image_positions()[0],
        'arrival_times': arrival_times,
        'cadence_mags': {b: [None]*num_images for b in bands} if cadence else None,
        'cadence_times': cadence,
        'num_pix': num_pix,
        'transform_matrix': transform_matrix
    }
    
    # Pre-calculate Lightcurves
    for band in bands:
        ps_mags = lens_class.point_source_magnitude(
            band=band, lensed=True, time=time_array, microlensing=True
        )
        for i in range(num_images):
            data['lightcurves'][band].append(ps_mags[0][i])
            
        if cadence is not None and band in cadence:
            cadence_time_array = np.array(cadence[band]) 
            ps_mags_cad = lens_class.point_source_magnitude(
                band=band, lensed=True, time=cadence_time_array, microlensing=True
            )
            for i in range(num_images):
                data['cadence_mags'][band][i] = ps_mags_cad[0][i]

    # Pre-calculate Normal Images & Normal RGBs
    for t in tqdm.tqdm(time_steps, desc="Simulating Base Frames"):
        current_frame_images = {}
        for band in bands:
            img = simulate_image(
                lens_class, num_pix=num_pix, band=band, 
                add_noise=True, observatory=get_observatory(band), 
                t_obs=t, 
            )
            data['images'][band].append(img)
            current_frame_images[band] = img
            
        rgb_list = [current_frame_images[b] for b in rgb_bands]
        rgb = rgb_image_from_image_list(image_list=rgb_list, stretch=rgb_stretch) 
        data['rgb_images'].append(rgb)
        
    # Pre-calculate ALL Difference Images & Difference RGBs to save time during animation
    print("Pre-computing Difference arrays...")
    for frame in range(len(time_steps)):
        if frame == 0:
            diff_rgb_list = [np.zeros_like(data['images'][b][0]) for b in rgb_bands]
            for band in bands:
                data['diff_images'][band].append(np.zeros_like(data['images'][band][0]))
        else:
            diff_rgb_list = []
            for band in bands:
                diff = data['images'][band][frame] - data['images'][band][0]
                data['diff_images'][band].append(diff)
                if band in rgb_bands:
                    diff_rgb_list.append(np.abs(diff))
                    
        diff_rgb = rgb_image_from_image_list(image_list=diff_rgb_list, stretch=rgb_stretch)
        data['diff_rgb_images'].append(diff_rgb)

    return data

# =====================================================================
# 2. Figure Setup
# =====================================================================
def setup_animation_figure(bands, num_images, data, rgb_bands, lens_class):
    """Sets up the Matplotlib GridSpec layout and initializes plot objects."""
    n_cols = len(bands) + 1 
    num_diff_plots = num_images - 1
    
    n_rows = 2 + num_images + 1 + num_diff_plots 
    hratios = [3, 3] + [1.5] * num_images + [0.5] + [1.5] * num_diff_plots
    
    fig = plt.figure(figsize=(2.5 * n_cols, sum(hratios) * 0.8)) 
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, height_ratios=hratios)
    
    # Extract lens and AGN parameters safely
    theta_E = lens_class.einstein_radius[0]
    z_s = lens_class.source_redshift_list[0]
    z_d = lens_class.deflector_redshift
    kappa_star, kappa_tot, shear, shear_angle = lens_class._microlensing_parameters_for_image_positions_single_source(band="i", source_index=0)
    
    try:
        log_m_bh = lens_class.source(0)._source._point_source.agn_class.kwargs_model['black_hole_mass_exponent']
        f_edd = lens_class.source(0)._source._point_source.agn_class.kwargs_model['eddington_ratio']
        agn_str = f" | $\\log_{{10}}(M_{{BH}}/M_\\odot)$: {log_m_bh:.2f} | $f_{{Edd}}$: {f_edd:.2f}"
    except Exception:
        agn_str = "" 
    
    fmt_arr = lambda arr: "[" + ", ".join([f"{x:.2f}" for x in arr]) + "]"
    title_str = (
        f"Lensed AGN Overview | Images: {num_images} | $\\theta_E$: {theta_E:.2f}\" | $z_d$: {z_d:.2f} | $z_s$: {z_s:.2f}{agn_str}\n"
        f"$\\kappa_*$: {fmt_arr(kappa_star)} | $\\kappa_{{tot}}$: {fmt_arr(kappa_tot)} | $\\gamma$: {fmt_arr(shear)}"
    )
    fig.suptitle(title_str, fontsize=14, weight='bold', y=0.98)
    
    plot_elements = {'images': {}, 'diff_images': {}, 'time_lines': []}
    
    def get_vlims(band):
        arr = np.array(data['images'][band])
        return np.min(arr), np.max(arr)

    # Row 0: Original Images
    for j, band in enumerate(bands):
        ax = fig.add_subplot(gs[0, j])
        vmin, vmax = get_vlims(band)
        im = ax.imshow(data['images'][band][0], cmap='gray', animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"{band}-band", fontsize=14)
        ax.axis('off')
        plot_elements['images'][band] = im
        if j == 0:
            ax.text(-0.15, 0.5, "Simulated\nImages", va='center', ha='right', transform=ax.transAxes, fontsize=14, weight='bold', rotation=90)

    # Row 0: RGB Image
    ax_rgb = fig.add_subplot(gs[0, -1])
    im_rgb = ax_rgb.imshow(data['rgb_images'][0], animated=True)
    ax_rgb.set_title(f"RGB ({rgb_bands[0]} {rgb_bands[1]} {rgb_bands[2]})", fontsize=14)
    ax_rgb.axis('off')
    
    x_pos, y_pos = data['positions']
    x_pos = (np.array(x_pos) / data['transform_matrix'][0, 0]) + data['num_pix'] // 2
    y_pos = (np.array(y_pos) / data['transform_matrix'][1, 1]) + data['num_pix'] // 2
    
    for i in range(num_images):
        ax_rgb.text(x_pos[i], y_pos[i], f"{i+1}", color='white', weight='bold', fontsize=8, ha='center', va='center')
    plot_elements['rgb'] = im_rgb

    # Row 1: Difference Images
    for j, band in enumerate(bands):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(data['diff_images'][band][0], cmap='coolwarm', animated=True, vmin=-1, vmax=1) 
        ax.set_title(f"Δ {band}-band", fontsize=12)
        ax.axis('off')
        plot_elements['diff_images'][band] = im
        if j == 0:
            ax.text(-0.15, 0.5, "Difference\nImages", va='center', ha='right', transform=ax.transAxes, fontsize=14, weight='bold', rotation=90)

    # Row 1: RGB Difference 
    ax_diff_rgb = fig.add_subplot(gs[1, -1])
    im_diff_rgb = ax_diff_rgb.imshow(data['diff_rgb_images'][0], animated=True)
    ax_diff_rgb.set_title(f"Δ RGB ({rgb_bands[0]} {rgb_bands[1]} {rgb_bands[2]})", fontsize=12)
    ax_diff_rgb.axis('off')
    plot_elements['diff_rgb'] = im_diff_rgb

    # Rows 2+: Normal Lightcurves
    colors = plt.cm.tab10.colors
    t_delay_ref = data['arrival_times'][0] 
    
    for i in range(num_images):
        ax_lc = fig.add_subplot(gs[2 + i, :])
        for j, band in enumerate(bands):
            ax_lc.plot(
                data['common_time'], data['lightcurves'][band][i], 
                label=f"{band}-band", color=colors[j], alpha=0.7, zorder=1
            )
            if data['cadence_times'] is not None and band in data['cadence_times']:
                ax_lc.scatter(
                    data['cadence_times'][band], data['cadence_mags'][band][i],
                    color=colors[j], marker='o', s=15, edgecolor='black', linewidth=0.5, zorder=2
                )
        
        vline = ax_lc.axvline(x=data['common_time'][0], color='black', linestyle='--', linewidth=2, zorder=3)
        plot_elements['time_lines'].append(vline)
        
        dt = data['arrival_times'][i] - t_delay_ref
        ax_lc.text(0.01, 0.85, f"Δt = {dt:.1f} days", transform=ax_lc.transAxes, fontsize=11, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        ax_lc.invert_yaxis()
        ax_lc.set_ylabel(f"Mag (Img {i+1})")
        ax_lc.set_xlim(data['common_time'][0], data['common_time'][-1])
        if i == 0:
            ax_lc.legend(loc='upper right')

    # Spacer Row
    ax_spacer = fig.add_subplot(gs[2 + num_images, :])
    ax_spacer.axis('off')
    ax_spacer.text(0.5, 0.5, "Difference Lightcurves ($\\Delta m = m_i - m_1$)", ha='center', va='center', fontsize=15, weight='bold', transform=ax_spacer.transAxes)

    # Rows (Spacer+): Difference Lightcurves
    for idx, i in enumerate(range(1, num_images)):
        row_idx = 2 + num_images + 1 + idx
        ax_diff = fig.add_subplot(gs[row_idx, :])
        
        for j, band in enumerate(bands):
            diff_lc = np.array(data['lightcurves'][band][i]) - np.array(data['lightcurves'][band][0])
            ax_diff.plot(
                data['common_time'], diff_lc, 
                color=colors[j], alpha=0.7, zorder=1
            )
            if data['cadence_times'] is not None and band in data['cadence_times']:
                diff_cad = np.array(data['cadence_mags'][band][i]) - np.array(data['cadence_mags'][band][0])
                ax_diff.scatter(
                    data['cadence_times'][band], diff_cad,
                    color=colors[j], marker='o', s=15, edgecolor='black', linewidth=0.5, zorder=2
                )

        vline = ax_diff.axvline(x=data['common_time'][0], color='black', linestyle='--', linewidth=2, zorder=3)
        plot_elements['time_lines'].append(vline)
        
        ax_diff.invert_yaxis()
        ax_diff.set_ylabel(f"ΔMag ({i+1} - 1)")
        ax_diff.set_xlim(data['common_time'][0], data['common_time'][-1])
        if idx == num_diff_plots - 1:
            ax_diff.set_xlabel("Observer Time (days)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, plot_elements

# =====================================================================
# 3. Animation Runner
# =====================================================================
def create_lensing_animation(lens_class, 
                             bands=["g", "r", "i", "z", "y"],
                             rgb_bands=["g", "r", "i"],
                             time_array=np.linspace(0, 30000, 300),
                             num_pix=41,
                             transform_matrix = np.array([[0.2, 0], [0, 0.2]]),
                             rgb_stretch=0.5,
                             cadence=None,
                             save_path=None,  # Rename to save_path to handle mp4
                             display_html=True): 
    
    data = generate_simulation_data(lens_class, time_array, bands, rgb_bands, 
                                    num_pix=num_pix, rgb_stretch=rgb_stretch,
                                    transform_matrix=transform_matrix, cadence=cadence)
                                    
    num_frames = len(data['image_times'][0])
    num_images = lens_class.image_number[0]
    
    fig, plot_elements = setup_animation_figure(bands, num_images, data, rgb_bands, lens_class)
    
    def update(frame):
        artists = []
        for band in bands:
            plot_elements['images'][band].set_array(data['images'][band][frame])
            plot_elements['diff_images'][band].set_array(data['diff_images'][band][frame])
            artists.append(plot_elements['images'][band])
            artists.append(plot_elements['diff_images'][band])
            
        plot_elements['rgb'].set_array(data['rgb_images'][frame])
        plot_elements['diff_rgb'].set_array(data['diff_rgb_images'][frame])
        artists.extend([plot_elements['rgb'], plot_elements['diff_rgb']])
        
        current_time = data['common_time'][frame]
        for vline in plot_elements['time_lines']:
            vline.set_xdata([current_time, current_time])
            artists.append(vline)
            
        return artists

    print("Compiling animation logic...")
    interval_ms = 80
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=True)
    
    if save_path is not None:
        print(f"Saving animation to: {save_path} ...")
        if save_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=int(1000/interval_ms), codec="h264")
            anim.save(save_path, writer=writer)
        else:
            writer = animation.PillowWriter(fps=int(1000/interval_ms))
            anim.save(save_path, writer=writer)
        print("Save complete!")

    plt.close(fig) 
    
    if display_html:
        print("Rendering HTML widget for notebook...")
        return HTML(anim.to_jshtml())
    else:
        print("Skipping HTML render. Animation is saved to disk!")
        return None


def plot_lightcurves(lens_class, bands, smooth_time_array, cadence_dict=None, microlensing=True, time_limits=None):
    """
    Plots continuous lightcurves (Col 1) and isolated microlensing residuals (Col 2), 
    overlaying cadence scatter points on both. 
    Colors map from blue to red based on the order of the provided `bands` list.
    
    :param time_limits: Tuple of (start_time, end_time) to restrict the x-axis. 
                        If provided, cadence observations are shifted to start at start_time.
    """
    num_images = lens_class.image_number[0]
    
    # ==========================================
    # 0. Time Limits & Cadence Shifting
    # ==========================================
    if time_limits is not None:
        t_start, t_end = time_limits
        
        # Filter the smooth array
        smooth_mask = (smooth_time_array >= t_start) & (smooth_time_array <= t_end)
        smooth_time_array = smooth_time_array[smooth_mask]
    else:
        t_start, t_end = smooth_time_array[0], smooth_time_array[-1]
        
    shifted_cadence = {}
    if cadence_dict is not None:
        active_bands = [b for b in bands if b in cadence_dict and len(cadence_dict[b]) > 0]
        if active_bands:
            min_cadence_time = np.min([np.min(cadence_dict[b]) for b in active_bands])
            
            for b in active_bands:
                # Shift so the global first observation lands on t_start
                shifted_times = np.array(cadence_dict[b]) - min_cadence_time + t_start
                
                # Filter to only keep cadence points within our plotting limits
                mask = (shifted_times >= t_start) & (shifted_times <= t_end)
                shifted_cadence[b] = shifted_times[mask]

    # Simple blue-to-red color mapping based on the order of the bands list
    cmap = plt.get_cmap('jet')
    band_colors = {band: cmap(i / max(1, len(bands) - 1)) for i, band in enumerate(bands)}
    
    # Create a 2-column grid
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 3 * num_images), sharex=True)
    
    if num_images == 1:
        axes = np.array([axes])
        
    for i in range(num_images):
        ax_main = axes[i, 0]  # Column 1: Total Lightcurve
        ax_resid = axes[i, 1] # Column 2: Microlensing Only
        
        for band in bands:
            current_color = band_colors[band]
            
            # ==========================================
            # 1. Smooth Lightcurves (The "Truth")
            # ==========================================
            smooth_mag_ml_true = lens_class.point_source_magnitude(
                band=band, lensed=True, time=smooth_time_array, microlensing=True
            )[0][i]
            
            smooth_mag_ml_false = lens_class.point_source_magnitude(
                band=band, lensed=True, time=smooth_time_array, microlensing=False
            )[0][i]
            
            smooth_mags = smooth_mag_ml_true if microlensing else smooth_mag_ml_false
            smooth_ml_residual = smooth_mag_ml_true - smooth_mag_ml_false
            
            ax_main.plot(smooth_time_array, smooth_mags, label=f"{band}-band (smooth)", 
                         color=current_color, alpha=0.5, zorder=1)
            
            ax_resid.plot(smooth_time_array, smooth_ml_residual, label=f"{band}-band (ML only)", 
                          color=current_color, alpha=0.5, zorder=1)
            
            # ==========================================
            # 2. Cadence Scatter Points (Interpolated)
            # ==========================================
            if band in shifted_cadence and len(shifted_cadence[band]) > 0:
                cadence_times = shifted_cadence[band]
                
                # INTERPOLATE instead of re-evaluating!
                # This guarantees they lie perfectly on the smooth curve and fixes the flat-line bug.
                cadence_mags = np.interp(cadence_times, smooth_time_array, smooth_mags)
                cadence_ml_residual = np.interp(cadence_times, smooth_time_array, smooth_ml_residual)
                
                ax_main.scatter(cadence_times, cadence_mags, color=current_color, 
                                marker='o', s=30, edgecolor='black', zorder=2, 
                                label=f"{band}-band (obs)")
                
                ax_resid.scatter(cadence_times, cadence_ml_residual, color=current_color, 
                                 marker='o', s=30, edgecolor='black', zorder=2)
                
        # ==========================================
        # 3. Formatting
        # ==========================================
        ax_main.invert_yaxis()
        ax_main.set_ylabel(f"Magnitude (Image {i+1})")
        ax_main.grid(True, alpha=0.3)
        if i == 0:
            ax_main.set_title("Total Lightcurve", weight='bold')
            
        ax_resid.invert_yaxis() 
        ax_resid.set_ylabel(r"$\Delta$ Mag (ML only)")
        ax_resid.grid(True, alpha=0.3)
        if i == 0:
            ax_resid.set_title("Microlensing Component", weight='bold')
            
    # Apply time limits to the x-axis
    axes[-1, 0].set_xlim(t_start, t_end)
    axes[-1, 1].set_xlim(t_start, t_end)
    axes[-1, 0].set_xlabel("Observer Time (days)")
    axes[-1, 1].set_xlabel("Observer Time (days)")
    
    # ==========================================
    # 4. Global Title and Legend Fixing
    # ==========================================
    fig.suptitle("Lensed Quasar Lightcurves", fontsize=16, weight='bold', y=0.98)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    
    fig.legend(by_label.values(), by_label.keys(), 
               loc='upper center', bbox_to_anchor=(0.5, 0.93), 
               ncol=min(len(bands), 6), frameon=True)
    
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    
    return fig