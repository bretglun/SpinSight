TOOLTIPS = {
    'load_presets':
    '# Load preset parameters\n'
    'Loads preset parameters typical for clinical sequences.',

    'object':
    '# Phantom object\n'
    'Selects the numerical phantom to be imaged. The phantom object determines the spatial distribution and MR properties of the simulated tissues.',

    'field_strength':
    '# B0 field strength\n'
    'Main magnetic field strength, *B₀*, measured in tesla (T). Field strength affects signal level, resonance frequencies, and tissue relaxation properties.',

    'parameter_style':
    '# Parameter style\n'
    'Determines which acquisition parameters are available, reflecting the styles of different vendors. Specifically if the receiver bandwidth is set in terms of the pixel bandwidth [Hz/pixel], the field of view (FOV) bandwidth [±kHz/FOV], or fat/water shift [pixels]; and if the spatial resolution is set in terms of matrix or voxel size.',

    'min_voxel_size':
    '# Minimum voxel size\n'
    'Minimum voxel size used internally to represent the simulated object. Smaller values provide greater spatial accuracy but require more memory and computation time when loading a phantom for the first time.',

    'noise_gain':
    '# Noise gain\n'
    'Controls the amount of simulated image noise. Increasing the noise gain reduces the signal-to-noise ratio (SNR).',

    'sequence_type':
    '# Pulse sequence\n'
    'Selects the type of MR pulse sequence. The sequence type is reflected in the pulse sequence diagram and determines the MR signal and which acquisition parameters are available.',

    'pixel_bandwidth':
    '# Pixel bandwidth\n'
    'Receiver bandwidth per acquired pixel in the frequency-encoding direction. Corresponds to the frequency range across a pixel when the readout gradient is switched on. A higher bandwidth shortens the signal readout, which reduces both the chemical shift displacement and the signal-to-noise ratio (SNR).',

    'FOV_bandwidth':
    '# FOV bandwidth\n'
    'Total receiver bandwidth across the field of view (FOV) in the frequency-encoding direction. Corresponds to the frequency range over the FOV when the readout gradient is switched on. A higher bandwidth shortens the signal readout, which reduces both the chemical shift displacement and the signal-to-noise ratio (SNR).',

    'FW_shift':
    '# Fat/water shift\n'
    'Displacement between fat and water in the frequency-encoding direction due to chemical shift. Fat/water whift is determined by the receiver bandwidth. Therefore, reducing the shift corresponds to increasing the bandwidth which results in shorter signal readouts and lower signal-to-noise ratio (SNR).',

    'NSA':
    '# Number of signal averages (NSA)\n'
    'Increasing the NSA improves the signal-to-noise ratio (SNR) by averaging repeated measurements, but increases scan time.',

    'partial_Fourier_factor':
    '# Partial Fourier factor\n'
    'Partiel Fourier imaging implies only a fraction of the *k*-space is acquired, such that one half is sampled fully, but the other only partially. The unmeasured *k*-space lines are synthesized from the measured based on the assumption that the *k*-space data is *conjugate symmetric*. Reducing the Partial Fourier factor enables shortening the scan time and the effective echo time if an EPI- or turbo spin echo train is used.',

    'turbo_factor':
    '# Turbo factor\n'
    'The spin echo train length, equal to the number of refocusing pulses per shot, which results in the same number of spin echoes. A turbo factor > 1 results in a turbo/fast spin echo sequence. Since multiple echoes are acquired, the number of shots required to sample *k*-space decreases, which reduces the scan time.',

    'EPI_factor':
    '# EPI factor\n'
    'The gradient echo train length, equal to the number of *k*-space lines acquired per gradient echo train. Since multiple echoes are acquired, the number of shots required to sample *k*-space decreases, which reduces the scan time. Since spin dephasing proceeds throughout the echo train, increasing the gradient echo train length results in more severe artifacts.\n\n'
    'Gradient echo trains are used in EPI sequences (Echo Planar Imaging).',

    'echo_order':
    '# Echo order\n'
    'Determines the order in which the *k*-space lines are acquired during a spin echo train.\n\n'
    '**Linear**: the *k*-space lines are acquired according to their position in *k*-space. This implies that central *k*-space is acquired in the middle of the echo train, which makes the shortest achievable effective echo time increase with echo train length. This scheme minimizes T₂-blurring.\n\n'
    '**Centric**: the *k*-space lines are acquired according to their distance from the *k*-space center. This implies that central *k*-space is acquired first, which minimizes the shortest achievable effective echo time.\n\n'
    '**Flexible**: the *k*-space lines are acquired to achieve the prescribed effective echo time while minimizing the echo train duration and avoiding T₂-blurring.',

    'FatSat':
    '# Fat saturation\n'
    '"FatSat" suppresses the signal from fat before the excitation pulse using a chemical-shift-selective RF pulse that tips the fat magnetization into the transverse plane, where it is dephased by a spoiler gradient, effectively destroying the fat signal.',

    'TR':
    '# Repetition time (TR)\n'
    'The time after which the core sequence (a shot) is repeated. TR determines how much time is available for T₁ relaxation (recovery of longitudinal magnetization) and therefore determines the degree of T₁-weighting. It also defines the duration of each shot and therefore determines the scan time.',

    'TE':
    '# Echo time (TE)\n'
    'The time between the excitation pulse and the acquired echo. During this time, T₂ relaxation (decay of transverse magnetization) occurs. Therefore, TE determines the degree of T₂-weighting (T₂*-weighting for spoiled gradient echo sequences).\n\n'
    'If multiple echoes are sampled, the *effective TE* corresponds to TE for the echo closest to the *k*-space center.',

    'TI':
    '# Inversion time (TI)\n'
    'The time between the opening 180° inversion pulse and the subsequent excitation pulse. After the inversion pulse, the now negative longitudinal magnetization relaxes through zero towards its positive equilibrium state, according to each tissue\'s T₁ relaxation time. Therefore, together with TR, TI determines the distinctive form of T₁ weighting characteristic of Inversion Recovery sequences. This includes the possibility of suppressing particular tissues on the basis of their T₁ relaxation time.',

    'FA':
    '# Flip angle\n'
    'Angle through which the net magnetization is rotated by the excitation RF-pulse. Together with TR, it influences signal strength and degree of T₁ weighting.',

    'trajectory':
    '# *k*-space trajectory\n'
    'Selects the path through *k*-space used to acquire the MR data.\n\n'
    '**Cartesian**: A rectangular *k*-space grid is sampled line by line.\n\n'
    '**Radial**: Lines passing through central *k*-space are measured at different angles to sample a star-shaped disc.\n\n'
    '**PROPELLER**: Paralell *k*-space lines are sampled in each shot, but the angle varies between shots so that each shot forms a "stripe" or "blade" through central *k*-space. (*PROPELLER = Periodically Rotated Overlapping ParallEL Lines with Enhanced Reconstruction*)\n\n',

    'frequency_direction':
    '# Frequency encoding direction\n'
    'Selects the spatial direction used for frequency encoding (i.e. in which the frequency/readout gradient is applied). The perpendicular in-plane direction is used for phase encoding.',

    'FOV_P':
    '# Field of view \n## (phase-encoding direction)\n'
    'Field of view (FOV) in the phase-encoding direction. It determines the spatial extent of the reconstructed image in this direction.',

    'FOV_F':
    '# Field of view \n## (frequency-encoding direction)\n'
    'Field of view (FOV) in the frequency-encoding direction. It determines the spatial extent of the reconstructed image in this direction.',

    'phase_oversampling':
    '# Phase oversampling\n'
    'Adds additional sampling in the phase-encoding direction beyond the prescribed field of view (FOV), so that the actually acquired FOV becomes larger than the prescribed / reconstructed FOV. This reduces wrap-around (aliasing) from anatomy outside the FOV, and increases the signal-to-noise ratio (SNR) at the cost of longer acquisition time.',

    'radial_sampling_factor':
    '# Radial sampling factor\n'
    'Number of acquired radial spokes or blades relative to full *k*-space coverage (Nyquist sampling limit). Increasing the radial sampling factor reduces streaking artifacts and increases signal-to-noise ratio (SNR) at the cost of longer acquisition times.',

    'matrix_P':
    '# Acquisition matrix\n## (phase-encoding direction)\n'
    'Nominal size of the acquisition matrix in the phase encoding direction, i.e. the number of phase-encoding steps before any phase over- or undersampling. Determines the spatial resolution in this direction together with the field of view (FOV).',

    'matrix_F':
    '# Acquisition matrix\n## (frequency-encoding direction)\n'
    'Number of acquired samples along the frequency-encoding direction. Determines the spatial resolution in this direction together with the field of view (FOV).',

    'voxel_P':
    '# Voxel size\n## (phase-encoding direction)\n'
    'Acquired voxel size in the phase-encoding direction, defining the spatial resolution in this direction. It also determines the acquisition matrix together with the field of view (FOV).',

    'voxel_F':
    '# Voxel size\n## (frequency-encoding direction)\n'
    'Acquired voxel size in the frequency-encoding direction, defining the spatial resolution in this direction. It also determines the acquisition matrix together with the field of view (FOV).',

    'recon_matrix_P':
    '# Reconstruction matrix\n## (phase-encoding direction)\n'
    'Number of pixels in the reconstructed image in the phase-encoding direction. If larger than the acquisition matrix, *zero filling* of *k*-space is used.',

    'recon_matrix_F':
    '# Reconstruction matrix\n## (frequency-encoding direction)\n'
    'Number of pixels in the reconstructed image in the frequency-encoding direction. If larger than the acquisition matrix, *zero filling* of *k*-space is used.',

    'recon_voxel_P':
    '# Reconstructed voxel size\n## (phase-encoding direction)\n'
    'Pixel size in the reconstructed image in the phase-encoding direction. If smaller than the acquired voxel size, *zero filling* of *k*-space is used.',

    'recon_voxel_F':
    '# Reconstructed voxel size\n## (frequency-encoding direction)\n'
    'Pixel size in the reconstructed image in the frequency-encoding direction. If smaller than the acquired voxel size, *zero filling* of *k*-space is used.',

    'slice_thickness':
    '# Slice thickness\n'
    'Prescribed thickness of the excited slice, defining the strength of the slice encoding gradient. Increasing the slice thickness improves the signal strength but reduces spatial resolution in the direction through the slice.',

    'radial_FOV_oversampling':
    '# Radial FOV oversampling factor\n'
    'Extends the sampled field of view (FOV) for radial imaging in the readout direction to reduce aliasing from signal outside the prescribed FOV.',

    'show_FOV':
    '# Show FOV\n'
    'Indicate the prescribed (yellow) and oversampled (light blue) field of view (FOV) on the image display.',

    'reference_tissue':
    '# Reference tissue\n'
    'Selects the tissue used as reference when calculating the relative signal-to-noise ratio (SNR).',

    'image_mode':
    '# Image mode\n'
    'Selects which component of the reconstructed complex MR image is displayed.\n\n'
    '**Magntiude**: The size of the transverse magnetization during sampling.\n\n'
    '**Phase**: The direction of the magnetization in the transverse plane during sampling.\n\n'
    '**Real**: The component of the transverse magnetization along one axis of the transverse plane.\n\n'
    '**Imaginary**: The component along the perpendicular axis.\n\n',

    'show_processed_kspace':
    '# Show processed *k*-space\n'
    'Displays *k*-space after processing rather than the originally measured *k*-space data.',

    'kspace_exponent':
    '# *k*-space exponent\n'
    'Applies nonlinear intensity scaling to the *k*-space display to make low-signal values more visible. Affects visualization only.',

    'kspace_mode':
    '# *k*-space mode\n'
    'Selects which component of the *k*-space data is displayed.\n\n'
    '**Magntiude**: The size of the sampled signal.\n\n'
    '**Phase**: The direction of the sampled signal in the transverse plane.\n\n'
    '**Real**: The component of the sampled signal along one axis of the transverse plane.\n\n'
    '**Imaginary**: The component along the perpendicular axis.\n\n',

    'homodyne':
    '# Homodyne\n'
    'Applies so-called *homodyne reconstruction* to synthesize missing *k*-space data in partial Fourier acquisitions.',

    'do_apodize':
    '# Apodization\n'
    'Applies a so-called *apodization filter* to *k*-space before image reconstruction to dampen the edge of the data matrix. This reduces Gibbs ringing artifact at the expense of some spatial resolution.',

    'apodization_alpha':
    '# Apodization alpha\n'
    'Controls the strength of the *k*-space apodization filter.',

    'do_zerofill':
    '# Zerofill\n'
    'Pad the *k*-space matrix with zeroes before reconstruction, according to the reconstructed matrix size. This results in smaller reconstructed pixels and reduces pixelization artifacts, but does not increase the spatial resolution.',

    'shot':
    '# Displayed shot\n'
    'Selects which acquisition shot is displayed in the pulse sequence diagram.',

    'signal_exponent':
    '# Signal exponent\n'
    'Applies nonlinear intensity scaling to the displayed signal to make low-signal values more visible. Affects visualization only.',
    }