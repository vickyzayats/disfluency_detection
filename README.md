# Disfluency detection code and pretrained model

The code to disfluency detection models: text-only baseline and late-fusion models.
The model configurations and parameters are specified in configs.py file.
Pretrained models will appear soon.

Training:
```
python train_disfl_with_attention_residual.py configs/attention_ms_word_level_total_phone_dur_normal_dist_text_only.py  num_feat 0
```

Inference:
```
python extract_probs_updated.py configs/attention_ms_word_level_total_phone_dur_normal_dist_text_only.py num_feat 0 model.h5
```

# Citation
If you use the code in your research, please cite:

Text and prosody model:

@article{zayats2019giving,
  title={Giving Attention to the Unexpected: Using Prosody Innovations in Disfluency Detection},
  author={Zayats, Vicky and Ostendorf, Mari},
  journal={Proc. NAACL},
  year={2019}
}

Text-only model:

@article{zayats2018robust,
  title={Robust cross-domain disfluency detection with pattern match networks},
  author={Zayats, Vicky and Ostendorf, Mari},
  journal={arXiv preprint arXiv:1811.07236},
  year={2018}
}
