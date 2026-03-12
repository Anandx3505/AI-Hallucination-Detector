@echo off
set PYTHON_EXE=C:\Users\Rishal\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo Installing standard requirements...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo.
echo Installing PyTorch Geometric from source...
"%PYTHON_EXE%" -m pip install torch_geometric
"%PYTHON_EXE%" -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

echo.
echo Environment setup complete.
