# Automate 


## NN Launcher
To be able to execute the `launcher_nn.py` script correctly, you need change in `script_nn.sh` in line 17 `PathToBeetorch` to your path from Desktop to the folder of the scripts.

Then you need to copy the files in `serverside` to a folder in the remote computers, and then change in `script_nn.sh` in line 17 `PathToScript` to your path from Desktop to the folder of yhe scripts.

Also in lines 30 and 43, add the path to the client side sh scripts.
