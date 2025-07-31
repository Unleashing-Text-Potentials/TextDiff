DATA_DIR=$1

#Video Data path
SUB_LINK_MSRVTT=""

#SUB_LINK_Didemo=
#SUB_LINK_Activity=
#SUB_LINK_LSMDC=

# Textdiff feature path
diff_image_msrvtt=""

#diff_image_lsmdc=""
#diff_image_anet=""
#diff_image_didemo=""

if [ -z $CUDA_VISIBLE_DEVICES ]; then
   CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
   --mount src=$(pwd),dst=/VidCLIP,type=bind \
   --mount src=$DATA_DIR,dst=/blob_mount,type=bind \
   --mount src=$SUB_LINK_MSRVTT,dst=/blob_mount/clip_data/vis_db/msrvtt_video_clips/videos_6fps,type=bind \
   --mount src=$diff_image_msrvtt,dst=/diff-image/msrvtt,type=bind \
   -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
   -w /VidCLIP tiankaihang/azureml_docker:horovod \
   bash -c "source /VidCLIP/setup.sh && export OMPI_MCA_btl_vader_single_copy_mechanism=none && bash"

#   --mount src=$SUB_LINK_Didemo,dst=/blob_mount/datasets/didemo/didemo_video_xfps,type=bind \
#   --mount src=$SUB_LINK_Activity,dst=/blob_mount/datasets/activitynet/ActivityNetVideoData2020Nov/video_frames_lr,type=bind \
#   --mount src=$SUB_LINK_LSMDC,dst=/blob_mount/datasets/lsmdc,type=bind \
#--mount src=$diff_image_anet,dst=/diff-image/anet,type=bind \
#   --mount src=$diff_image_didemo,dst=/diff-image/didemo,type=bind \
#   --mount src=$diff_image_lsmdc,dst=/diff-image/lsmdc,type=bind \