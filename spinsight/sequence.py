import numpy as np
from spinsight import constants
import scipy.signal as signal
import scipy.interpolate as interpolate


def accumulateSlopes(slopes):
    # slopes shall be a list of (t, dG, dt) tuples where t is starttime and dG is the gradient change over duration dt
    slopes = sorted(slopes, key=lambda x: x[0]) # sort by time
    waveform = []
    slewing = []
    G = 0
    for (t, dG, dt) in slopes:
        keepSlewing = []
        for (slew, stoptime) in slewing:
            G += slew * (min(stoptime, t) - waveform[-1][0])
            if stoptime > t:
                keepSlewing.append((slew, stoptime))
        slewing = list(keepSlewing)
        if len(waveform)>2 and (t-waveform[-2][0]) < 1e-6: # nanosecond tolerance
            waveform.pop(-1) # remove intermediate gradient amplitudes at an instant
        waveform.append((t, G)) # (time, gradient)
        if dG != 0:
            if dt > 0:
                slewing.append((dG/dt, t + dt)) # (slew, stoptime)
            else: # infinite slope
                G += dG
    return waveform


def accumulateWaveforms(waveforms, board):
    slopes = []
    for waveform in waveforms:
        slopes += zip(waveform['time'], np.diff(waveform[board], append=0), np.diff(waveform['time'], append=waveform['time'][-1]))
    return accumulateSlopes(slopes)


def prepareWaveform(waveform, t0, t1, scale=1):
    wf = np.concatenate(([0], np.array(waveform), [0])) * scale
    t = np.concatenate(([t0], np.linspace(t0, t1, len(waveform)), [t1]))
    return wf, t


def getFWHM(s, t):
    S = np.abs(np.fft.fftshift(np.fft.fft(s)))
    f = np.fft.fftshift(np.fft.fftfreq(len(t)) / (t[1]-t[0]) * 1e3)
    spline = interpolate.UnivariateSpline(f, S-np.max(S)/2, s=0)
    f1, f2 = spline.roots() # find the roots
    return abs(f2-f1)


def getRF(flipAngle, dur, name, time=0., shape='hammingSinc'):
    match shape:
        case 'rect':
            waveform = np.array([1., 1.])
        case 'hammingSinc':
            n = 51
            waveform = np.sinc((np.arange(n)-n/2)/n*5) * signal.windows.hamming(n)
        case _:
            raise NotImplementedError(shape)

    t0, t1 = time - dur/2, time + dur/2

    scale = flipAngle / (np.mean(waveform) * dur / 1e3 * constants.GYRO * 360)

    am, t = prepareWaveform(waveform, t0, t1, scale)
    rf = { 'RF': am,
            'time': t,
            'name': name,
            'center': '{:.1f} ms'.format(time),
            'center_f': time,
            'duration': '{:.1f} ms'.format(dur),
            'dur_f': dur,
            'flip_angle': '{:.0f}°'.format(flipAngle),
            'FWHM_f': getFWHM(am[1:-1], t[1:-1])}
    return rf


def getGradient(dir, time=0., maxAmp=25., maxSlew=80., totalArea=None, flatArea=None, flatDur=None, name=''):
    assert(sum([x is not None for x in [totalArea, flatArea, flatDur]])==1)
    if totalArea is not None:
        slewArea = maxAmp**2 / maxSlew
        if abs(totalArea)<slewArea:
            maxAmp = np.sqrt(abs(totalArea)*maxSlew) * np.sign(totalArea)
            flatDur = 0.
        else:
            flatArea = (abs(totalArea)-slewArea) * np.sign(totalArea)
    if flatArea is not None:
        flatDur = abs(flatArea / maxAmp)
        maxAmp = abs(maxAmp) * np.sign(flatArea)
    riseTime = abs(maxAmp)/maxSlew
    amp = np.array([0., maxAmp, maxAmp, 0.])
    t = np.cumsum(np.array([0., riseTime, flatDur, riseTime]))
    dur = t[-1]-t[0]
    t += time - dur/2
    area = getGradientArea(amp, t)
    gr = {
        dir: amp,
        'time': t,
        'name': name,
        'center': '{:.1f} ms'.format(time),
        'center_f': time,
        'duration': '{:.1f} ms'.format(dur),
        'dur_f': dur,
        'flatDur_f': flatDur,
        'riseTime_f': riseTime,
        'area': '{:.1f} μTs/m'.format(area),
        'area_f': area
    }
    return gr


def getGradientArea(g, t):
    return sum(np.diff(t) * (g[:-1] + g[1:]))/2


def moveWaveform(wf, time):
    oldTime = wf['center_f']
    wf['time'] += time - oldTime
    wf['center'] = '{:.2f} ms'.format(time)
    wf['center_f'] = time


def rescaleGradient(g, scale):
    for dir in ['slice', 'phase', 'frequency']:
        if dir in g:
            g[dir] *= scale
    g['area_f'] *= scale
    g['area'] = '{:.2f} μTs/m'.format(g['area_f']),


def getADC(dur, name, time=0.):
    adc = {
        'name': name,
        'time': np.array([-dur/2, dur/2]) + time,
        'center_f': time,
        'dur_f': dur
    }
    return adc