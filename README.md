# Paper

**ICIC2024: Open-Domain Question Answering over Tables with Large Language Models**

![image](image/your_image.png)

## Environment
- Python 3.7
- PyTorch 1.8

## Dataset Download
- [OTT-QA](https://opendatalab.org.cn/OpenDataLab/OTT-QA)
- [NQ-TABLES](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md)

## Generate Identifiers
The main code is in `key_word.py`.

## Retriever
The main code is in `retriever/embedding.py`.

Re-ranking: `BM25.py`

## Selector
- `link.py`
- `sub_choose.py`
- `sub_choose2.py`

## Reader
`run_qa.py`

## Evaluation
`evaluate_script.py`

## Notes
Due to the need for certain keys and data loss, the project code is currently not runnable. If you want to understand the general steps, you can review the code or browse the provided example images in the `images/` directory.

## Citations
```text
@inproceedings{liang2024open,
  title={Open-Domain Question Answering over Tables with Large Language Models},
  author={Liang, Xinyi and Hu, Rui and Liu, Yu and Zhu, Konglin},
  year={2024},
  organization={Springer}
}
```
