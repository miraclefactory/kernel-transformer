# Kernel Transformer
A transformer model based on sliding kernel self attention mechanism. This model is based on a implementation of Swin Transformer. See [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer) for the original implementation.

### Some preliminary comparisons on CIFAR10
| Model | Params | Val. Acc. |
| :---: | :---: | :---: |
| Swin Transformer (tiny) | 26,598,166 | 85.3% |
| Kernel Transformer (tiny) | 26,600,362 | 85.9% |

<img width="598" alt="kernel_vs_swin" src="https://github.com/miraclefactory/kernel-transformer/assets/89094576/4d5581c2-bc09-4ac0-bd23-6f188dd011f1">

