# MV-VLM
Implementation for the screen time project using multi-view vision language model https://arxiv.org/abs/2410.01966

You need to download the Llama model from https://huggingface.co/meta-llama/Llama-2-7b.



# Data preprocess
You need to install unsloth environment to generate the text embedding.
https://github.com/unslothai/unsloth.git

```
python step1_build_group_caption_example_caption.py
python step2_convert_group_caption_into_emb.py
```

# Run the model
```
conda create --name MVVLM --file requirements.txt
cd MVVLM
conda activate MVVLM
bash scripts/run.sh
```

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
