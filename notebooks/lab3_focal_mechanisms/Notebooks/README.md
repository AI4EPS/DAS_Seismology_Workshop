# In terminal

```bash
cd /home/jovyan/workshop-repo
git pull origin main
conda init
source ~/.bashrc
conda create -n dasfm python=3.11 -y
conda activate dasfm
conda install -c conda-forge ipykernel -y
python -m ipykernel install --user --name dasfm --display-name dasfm
```

# In Jupyter Notebook Select kernel dasfm

# In cell (only need to do this once):

```python
%pip install torch --index-url https://download.pytorch.org/whl/cu126
%pip install -e /home/jovyan/workshop-repo/notebooks/lab3_focal_mechanisms/Scripts/dasfm_workshop
```
