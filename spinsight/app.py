import holoviews as hv
import panel as pn
from pathlib import Path
import toml
from spinsight import MRIsimulator
from functools import partial
from datetime import datetime


def hideShowButtonCallback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


def loadButtonCallback(simulator, settingsFile, event):
    print('Loading settings from file', settingsFile)
    with open(settingsFile, 'r') as f:
        settings = toml.load(f)
    simulator.setParams(settings)


def saveButtonCallback(simulator, settingsFile, event):
    print('Saving settings to file', settingsFile)
    settings = simulator.getParams()
    with open(settingsFile, 'w') as f:
        toml.dump(settings, f)


def infoNumber(name, value, format, textColor):
    return pn.indicators.Number(default_color=textColor, name=name, format=format, font_size='12pt', title_size='12pt', value=value)


def infoString(name, value, textColor):
    return pn.indicators.String(default_color=textColor, name=name, font_size='12pt', title_size='12pt', value=value)


def getApp(darkMode=True, settingsFilestem='', startTime=datetime.now(), lazySliders=False):
    pn.config.theme = 'dark' if darkMode else 'default'
    pn.config.throttled = lazySliders
    textColor = 'white' if darkMode else 'black' # needed for pn.indicators.Number which doesn't respect pn.config.theme

    settingsFile = Path(settingsFilestem).with_suffix('.toml') if bool(settingsFilestem) else Path('')

    simulator = MRIsimulator.MRIsimulator(name='')
    title = 'SpinSight MRI simulator'
    author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
    try:
        version = f', v {toml.load(Path(__file__).parent.parent / "pyproject.toml")["project"]["version"]}'
    except (FileNotFoundError, KeyError):
        version = ''
    
    discreteSliderParams = ['TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'phaseOversampling', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'voxelF', 'voxelP', 'reconVoxelF', 'reconVoxelP', 'sliceThickness', 'pixelBandWidth', 'FOVbandwidth', 'FWshift', 'EPIfactor']
    paramPanels = {name: pn.panel(simulator.param, parameters=params, widgets={p: pn.widgets.DiscreteSlider for p in params if p in discreteSliderParams}, name=name) for name, params in [
        ('Settings', ['object', 'fieldStrength', 'parameterStyle']),
        ('Contrast', ['FatSat', 'TR', 'TE', 'FA', 'TI']),
        ('Geometry', ['trajectory', 'frequencyDirection', 'FOVF', 'FOVP', 'phaseOversampling', 'radialFactor', 'voxelF', 'voxelP', 'matrixF', 'matrixP', 'reconVoxelF', 'reconVoxelP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness']),
        ('Sequence', ['sequence', 'pixelBandWidth', 'FOVbandwidth', 'FWshift', 'NSA', 'partialFourier', 'turboFactor', 'EPIfactor']),
        ('Post-processing', ['homodyne', 'doApodize', 'apodizationAlpha', 'doZerofill']),
    ]}

    infoPane = pn.Row(infoNumber(name='Relative SNR', format='{value:.0f}%', value=simulator.param.relativeSNR, textColor=textColor),
                      infoString(name='Scan time', value=simulator.param.scantime, textColor=textColor),
                      infoNumber(name='Fat/water shift', format='{value:.2f} pixels', value=simulator.param.FWshift, textColor=textColor),
                      infoNumber(name='Bandwidth', format='{value:.0f} Hz/pixel', value=simulator.param.pixelBandWidth, textColor=textColor))
    shotAngleInfo = infoNumber(name='Angle', format='{value:.0f}°', value=simulator.param.spokeAngle, textColor=textColor) 
    numShotsInfo = infoNumber(name='# shots', format='{value:.0f}', value=simulator.param.num_shots, textColor=textColor)
    def update_num_shots_label(event): 
        numShotsInfo.name = event.new
    simulator.param.watch(update_num_shots_label, 'num_shots_label')
    
    dmapKspace = pn.Column(hv.DynamicMap(simulator.getKspace) * simulator.kLine, 
                           # simulator.param.kspaceType, 
                           pn.Row(simulator.param.showProcessedKspace, simulator.param.kspaceExponent), 
                           visible=False)
    dmapMRimage = hv.DynamicMap(simulator.getImage)
    dmapSequence = pn.Column(hv.DynamicMap(simulator.getSequencePlot), pn.Row(simulator.param.shot, shotAngleInfo, numShotsInfo), visible=False)
    loadButton = pn.widgets.Button(name='Load settings', visible=settingsFile.is_file())
    loadButton.on_click(partial(loadButtonCallback, simulator, settingsFile))
    saveButton = pn.widgets.Button(name='Save settings', visible=settingsFile.is_file())
    saveButton.on_click(partial(saveButtonCallback, simulator, settingsFile))
    sequenceButton = pn.widgets.Button(name='Show sequence')
    sequenceButton.on_click(partial(hideShowButtonCallback, dmapSequence))
    kSpaceButton = pn.widgets.Button(name='Show k-space')
    kSpaceButton.on_click(partial(hideShowButtonCallback, dmapKspace))
    resetSNRbutton = pn.widgets.Button(name='Set reference SNR')
    resetSNRbutton.on_click(simulator.setReferenceSNR)
    dashboard = pn.Column(
        pn.Row(
            pn.Column(
                pn.Row(
                    pn.Column(
                        pn.Row(sequenceButton, kSpaceButton), 
                        paramPanels['Sequence'],
                        paramPanels['Contrast']
                    ), 
                    pn.Column(
                        paramPanels['Geometry']
                    )
                )
            ), 
            pn.Column(
                dmapMRimage, 
                pn.Column(
                    # simulator.param.imageType, 
                    pn.Row(resetSNRbutton, simulator.param.showFOV), 
                    simulator.param.referenceTissue, 
                    infoPane
                )
            ), 
            pn.Column(
                dmapKspace,
                paramPanels['Post-processing']
            )
        ), 
        dmapSequence, 
        pn.Column(pn.pane.Markdown(author),
                  pn.pane.Markdown(f'*(server started {startTime: %Y-%m-%d %H:%M:%S}{version})*', styles={'color': 'gray'}),
                  height=10),
        sizing_mode='stretch_both'
    )

    template = pn.template.FastListTemplate(
        title=title,
        theme='dark' if darkMode else 'default',
        theme_toggle=False,
        header_background='#262626',
        modal=[pn.Column()],
        collapsed_sidebar=True
    )
    template.main.append(dashboard)
    template.sidebar.append(
        pn.Column(
            pn.Row(loadButton, saveButton), 
            paramPanels['Settings'], 
        )
    )
    return template
