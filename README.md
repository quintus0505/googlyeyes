# GOOGLYEYES

Rename the Original libstdc++.so.6:

To create a backup, rename the existing libstdc++.so.6 file by adding _bak to the name:
bash
复制代码
mv libstdc++.so.6 libstdc++.so.6_bak
Create the Symlink to the System Version:

Now, link the system’s libstdc++.so.6 to your environment:
bash
复制代码
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/googlyeyes/lib/libstdc++.so.6