# PyInstaller spec for Reach all-in-one executable
# Build: pyinstaller reach.spec

a = Analysis(
    ['reach-core/src/reach/cli.py'],
    pathex=['reach-core/src'],
    binaries=[],
    datas=[('reach-server/templates/index.html', 'reach-server/templates')],
    hiddenimports=['reach', 'reach.cli', 'reach.models', 'reach.extract', 'reach.tagtree', 'reach.construct', 'reach.validate', 'reach.enrich'],
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
