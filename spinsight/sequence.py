import numpy as np
from spinsight import constants
import scipy.signal as signal
import scipy.interpolate as interpolate


def accumulate_slopes(slopes):
    # slopes shall be a list of (t, dG, dt) tuples where t is starttime and dG is the gradient change over duration dt
    slopes = sorted(slopes, key=lambda x: x[0]) # sort by time
    waveform = []
    slewing = []
    G = 0
    for (t, dG, dt) in slopes:
        keep_slewing = []
        for (slew, stoptime) in slewing:
            G += slew * (min(stoptime, t) - waveform[-1][0])
            if stoptime > t:
                keep_slewing.append((slew, stoptime))
        slewing = list(keep_slewing)
        if len(waveform)>2 and (t-waveform[-2][0]) < 1e-6: # nanosecond tolerance
            waveform.pop(-1) # remove intermediate gradient amplitudes at an instant
        waveform.append((t, G)) # (time, gradient)
        if dG != 0:
            if dt > 0:
                slewing.append((dG/dt, t + dt)) # (slew, stoptime)
            else: # infinite slope
                G += dG
    return waveform


def accumulate_waveforms(waveforms, board):
    slopes = []
    for waveform in waveforms:
        slopes += zip(waveform['time'], np.diff(waveform[board], append=0), np.diff(waveform['time'], append=waveform['time'][-1]))
    return accumulate_slopes(slopes)


def prepare_waveform(waveform, t0, t1, scale=1):
    wf = np.concatenate(([0], np.array(waveform), [0])) * scale
    t = np.concatenate(([t0], np.linspace(t0, t1, len(waveform)), [t1]))
    if t0 < t1:
        return wf, t
    else:
        return np.flip(wf), np.flip(t)


def get_FWHM(s, t):
    S = np.abs(np.fft.fftshift(np.fft.fft(s)))
    f = np.fft.fftshift(np.fft.fftfreq(len(t)) / (t[1]-t[0]) * 1e3)
    spline = interpolate.UnivariateSpline(f, S-np.max(S)/2, s=0)
    f1, f2 = spline.roots() # find the roots
    return abs(f2-f1)


def get_RF(flip_angle, dur, name, time=0., shape='hamming_sinc'):
    match shape:
        case 'rect':
            waveform = np.array([1., 1.])
        case 'hamming_sinc':
            n = 51
            waveform = np.sinc((np.arange(n)-n/2)/n*5) * signal.windows.hamming(n)
        case _:
            raise NotImplementedError(shape)

    t0, t1 = time - dur/2, time + dur/2

    scale = flip_angle / (np.mean(waveform) * dur / 1e3 * constants.GYRO * 360)

    am, t = prepare_waveform(waveform, t0, t1, scale)
    rf = {'RF': am,
          'time': t,
          'name': name,
          'center': f'{time:.1f} ms',
          'center_f': time,
          'duration': f'{dur:.1f} ms',
          'dur_f': dur,
          'flip_angle': f'{flip_angle:.0f}°',
          'FWHM_f': get_FWHM(am[1:-1], t[1:-1])}
    return rf


def get_signal(signal, time, scale=1.0, exponent=1.0, name='sampling'):
    t0, t1 = time[0], time[-1]
    am, t = prepare_waveform(signal, t0, t1, scale)
    am = np.sign(am) * np.abs(am)**exponent
    signal = {'signal': am,
              'time': t,
              'name': name,
              'center': f'{(t0+t1)/2:.1f} ms',
              'duration': f'{abs(t1-t0):.1f} ms'}
    return signal


def get_gradient(dir, time=0., max_amp=25., max_slew=80., total_area=None, flat_area=None, flat_dur=None, waveform=None, name=''):
    assert(sum([x is not None for x in [total_area, flat_area, flat_dur]])==1)
    if total_area is not None:
        slewArea = max_amp**2 / max_slew
        if abs(total_area)<slewArea:
            max_amp = np.sqrt(abs(total_area)*max_slew) * np.sign(total_area)
            flat_dur = 0.
        else:
            flat_area = (abs(total_area)-slewArea) * np.sign(total_area)
    if flat_area is not None:
        flat_dur = abs(flat_area / max_amp)
        max_amp = abs(max_amp) * np.sign(flat_area)
    amp = np.array(waveform) if waveform is not None else np.array([max_amp, max_amp])
    risetime = max(abs(amp[0]), abs(amp[-1]))/max_slew
    t = np.cumsum(np.array([0., risetime] + [flat_dur/(len(amp)-1)]*(len(amp)-1) + [risetime]))
    amp = np.pad(amp, (1, 1), mode='constant', constant_values=0)    
    dur = t[-1]-t[0]
    t += time - dur/2
    area = get_gradient_area(amp, t)
    gr = {
        dir: amp,
        'time': t,
        'name': name,
        'center': f'{time:.1f} ms',
        'center_f': time,
        'duration': f'{dur:.1f} ms',
        'dur_f': dur,
        'flat_dur_f': flat_dur,
        'risetime_f': risetime,
        'area': f'{area:.1f} μTs/m',
        'area_f': area
    }
    return gr


def get_gradient_area(g, t):
    return sum(np.diff(t) * (g[:-1] + g[1:]))/2


def move_waveform(wf, time):
    old_time = wf['center_f']
    wf['time'] += time - old_time
    wf['center'] = f'{time:.1f} ms'
    wf['center_f'] = time


def rescale_gradient(g, scale):
    for dir in ['slice', 'phase', 'frequency']:
        if dir in g:
            g[dir] *= scale
    g['area_f'] *= scale
    g['area'] = f'{g["area_f"]:.1f} μTs/m',


def get_ADC(dur, name, time=0.):
    adc = {
        'name': name,
        'time': np.array([-dur/2, dur/2]) + time,
        'center': f'{time:.1f} ms',
        'center_f': time,
        'duration': f'{dur:.1f} ms',
        'dur_f': dur
    }
    return adc