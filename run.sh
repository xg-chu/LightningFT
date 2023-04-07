# python run_track.py -d 7 --data path.mp4 -v
# python run_track.py -d 7 --data path.mp4 --base ./outputs/path 
# python build_dataset.py --train ./outputs/path --test ./outputs/path
# python build_dataset.py --add_bg ~/workspace/Data/nerface_dataset/person_1/bg/00050.png

target_path="/cto_studio/vistring/liuyunfei/repo/dataset/HDTF-raw"
videos=$(ls $target_path)

for file in $videos
do
    file_path="${target_path}/${file}"
    if [[ $file_path == *".mp4"* ]]; then
        echo $file_path
    fi
done

for file in $videos
do
    file_path="${target_path}/${file}"
    if [[ $file_path == *".mp4"* ]] && ! [[ $file_path == *"RD_Radio"* ]]; then
        echo "python track_lightning.py -d 7 --data $file_path -v --synthesis"
        python track_lightning.py -d 7 --data $file_path -v --synthesis
    fi
done
