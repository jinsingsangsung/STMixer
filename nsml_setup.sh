cd /mnt/video-nfs5/datasets/ava
cp -r frames.tar /mnt/tmp
cp -r annotations /mnt/tmp
cp -r frame_lists /mnt/tmp
cd /mnt/tmp
cp -r ./annotations/ava_train_v2.2_1000.csv ./frame_lists/
tar -xf frames.tar
rm -rf frames.tar
cd ~/STMixer
python3 ./make_mini_csv.py