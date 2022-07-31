# Let's make sure the kaggle.json file is present.
ls -lha kaggle.json
# Next, install the Kaggle API client.
pip install -q kaggle
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
# This permissions change avoids a warning on Kaggle tool startup.
chmod 600 ~/.kaggle/kaggle.json