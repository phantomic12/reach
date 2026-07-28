# PyInstaller spec for Reach all-in-one executable
# Build: pyinstaller reach.spec
#
# Hidden imports cover every reachable submodule from reach.* plus the heavy
# ML/OCR backends (Unlimited OCR + transformers + einops + addict + easydict +
# doclayout-yolo + surya). Without these, PyInstaller's static analysis misses
# dynamically-imported modules and the bundled binary crashes at runtime with
# ModuleNotFoundError when the user actually runs remediation.

a = Analysis(
    ['reach-core/src/reach/cli.py'],
    pathex=['reach-core/src'],
    binaries=[],
    datas=[('reach-server/templates/index.html', 'reach-server/templates')],
    hiddenimports=[
        # Reach package surface
        'reach',
        'reach.cli',
        'reach.models',
        'reach.extract',
        'reach.tagtree',
        'reach.construct',
        'reach.validate',
        'reach.enrich',
        # OCR backend (Unlimited OCR via transformers)
        'transformers',
        'transformers.dynamic_module_utils',
        'transformers.models.auto',
        'transformers.models.auto.tokenization_auto',
        'transformers.models.auto.configuration_auto',
        'transformers.models.auto.auto_factory',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.processing_auto',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_base',
        'transformers.configuration_utils',
        'transformers.modeling_utils',
        'transformers.utils',
        'transformers.utils.import_utils',
        'transformers.generation',
        'transformers.generation.streamers',
        # OCR third-party deps
        'einops',
        'einops._backends',
        'einops._torch_specific',
        'addict',
        'easydict',
        # Layout backends
        'doclayout_yolo',
        'surya',
        'surya.inference',
        'surya.layout',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='reach',
    console=True,
    icon=None,
)
