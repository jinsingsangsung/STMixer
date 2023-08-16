cd /mnt/video-nfs5/datasets/ava
cp -r frames.tar /mnt/tmp
cp -r annotations /mnt/tmp
cp -r frame_lists /mnt/tmp
cd /mnt/tmp
tar -xf frames.tar
rm -rf frames.tar
cd ~/STMixer