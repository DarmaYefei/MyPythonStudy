# -*- mode: python -*-

block_cipher = None


a = Analysis(['D:\\MyPythonDoc\\MyPythonStudy\\MyPythonStudy\\PythonCharm\\Game2048\\python_2048.py'],
             pathex=['D:\\MyPythonDoc\\MyPythonStudy\\MyPythonStudy\\PythonCharm'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='python_2048',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
