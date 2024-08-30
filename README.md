# Phi-3-DPO
a Phi-3 DPO training script. 一个用于Phi-3 DPO训练的脚本。
## Start
To start DPO training for Phi-3 model, 运行`dpo.py`即可
```
python dpo.py
```
or start DPO training with an evaluation of cross entropy loss on another eval dataset, run 假如想增加一个测试集求取交叉熵损失函数，运行`dpo-eval-another-loss.py`即可
```
python dpo-eval-another-loss.py
```
