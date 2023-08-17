import csv
import pdb


ref = []
with open('/mnt/tmp/frame_lists/ava_train_v2.2_1000.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
       ref.append(row)

header = ["original_vido_id", "video_id", "frame_id", "path", "labels"]
with open('/mnt/tmp/frame_lists/train_1000.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=" ", quotechar = "'") 
    writer.writerow(header)
    video_i = -1
    prev_vid = ""
    prev_fid = -1
    for entry in ref:
        vid = entry["original_video_id"]
        fid = entry["frame_id"]
        if prev_vid != vid:
            video_i += 1
            prev_vid = vid
        if prev_fid == fid:
            continue
        fmt_fid = f"{int(fid):06d}"
        video_path = f"{vid}/{vid}_{fmt_fid}.jpg"
        # line = f'{vid} {video_i} {fid} {video_path} ""'
        writer.writerow([vid, video_i, fid, video_path, '""'])
            
import pdb; pdb.set_trace()