# Kernel Transformer
A transformer model based on sliding kernel self attention mechanism. This model is based on a implementation of Swin Transformer. See [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer) for the original implementation.

### Preliminary comparisons on CIFAR10
| Model | Params | Val. Acc. |
| :---: | :---: | :---: |
| Swin Transformer (tiny) | 26,598,166 | 85.3% @200eps |
| Swin Transformer (tiny) | 26,598,166 | 85.8% @300eps |
| Kernel Transformer (tiny) | 26,600,362 | 85.9% @200eps |
| Kernel Transformer (tiny) | 26,600,362 | 87.1% @300eps |
| Kernel Transformer (super-tiny) | 13,637,866 | 84.7% @300eps |

<img width="598" alt="kernel_vs_swin" src="https://github.com/miraclefactory/kernel-transformer/assets/89094576/4d5581c2-bc09-4ac0-bd23-6f188dd011f1">

