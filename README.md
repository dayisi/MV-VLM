# MV-VLM
This is the implementation for the screen time project using a multi-view vision language model https://arxiv.org/abs/2410.01966

You need to download the [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

# Data preprocess

You need to install unsloth environment to use the MiniLM to generate text embedding.

```
pip install unsloth
python step1_build_group_caption_example_caption.py
python step2_convert_group_caption_into_emb.py
```

# Run the model

The checkpoint file can be downloaded via the link: [checkpoint_epoch5_step437.pth](https://stevens0-my.sharepoint.com/:u:/g/personal/xhou11_stevens_edu/ETo_NSJyX19Lujtely1VqBEBgFHitXBFqsUb_z0AYloR4w?e=iZAjdZ).

Please download it and save it into the ```MVVLM/saved_ckpt``` folder.

```
conda create --name MVVLM --file requirements.txt
cd MVVLM
conda activate MVVLM
bash scripts/run.sh
```

# Postprocessing
After the screen type identification, ChatGPT4 will be utilized to smooth the description for scene understanding.

# Data availability
The dataset used in the paper is not publicly available due to privacy and IRB restrictions.

# Citation

If you use this code, please cite our paper:
```bibtex
@article{hou2024enhancing,
  title={Enhancing Screen Time Identification in Children with a Multi-View Vision Language Model and Screen Time Tracker},
  author={Hou, Xinlong and Shen, Sen and Li, Xueshen and Gao, Xinran and Huang, Ziyi and Holiday, Steven J and Cribbet, Matthew R and White, Susan W and Sazonov, Edward and Gan, Yu},
  journal={arXiv preprint arXiv:2410.01966},
  year={2024}
}
```
