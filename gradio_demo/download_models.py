from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="yisol/IDM-VTON", filename="densepose/model_final_162be9.pkl", local_dir="/home/jupyter/trials/IDM-VTON/ckpt/densepose/")
hf_hub_download(repo_id="yisol/IDM-VTON", filename="humanparsing/parsing_atr.onnx", local_dir="/home/jupyter/trials/IDM-VTON/ckpt/humanparsing")
hf_hub_download(repo_id="yisol/IDM-VTON", filename="humanparsing/parsing_lip.onnx", local_dir="/home/jupyter/trials/IDM-VTON/ckpt/humanparsing")
hf_hub_download(repo_id="yisol/IDM-VTON", filename="openpose/ckpts/body_pose_model.pth", local_dir="/home/jupyter/trials/IDM-VTON/ckpt/openpose/ckpts/")

# https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt/openpose/ckpts

# https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl
