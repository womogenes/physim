## Things to try

- Transformer is weird

```
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
ParticleTransformer                           [2, 512, 4]               --
├─Sequential: 1-1                             [1024, 64]                --
│    └─Linear: 2-1                            [1024, 64]                384
│    └─ReLU: 2-2                              [1024, 64]                --
│    └─Dropout: 2-3                           [1024, 64]                --
│    └─Linear: 2-4                            [1024, 64]                4,160
│    └─ReLU: 2-5                              [1024, 64]                --
│    └─Dropout: 2-6                           [1024, 64]                --
├─TransformerEncoder: 1-2                     [2, 512, 64]              --
│    └─ModuleList: 2-7                        --                        --
│    │    └─TransformerEncoderLayer: 3-1      [2, 512, 64]              83,008
│    │    └─TransformerEncoderLayer: 3-2      [2, 512, 64]              83,008
│    │    └─TransformerEncoderLayer: 3-3      [2, 512, 64]              83,008
│    │    └─TransformerEncoderLayer: 3-4      [2, 512, 64]              83,008
├─Linear: 1-3                                 [2, 512, 4]               260
===============================================================================================
Total params: 336,836
Trainable params: 336,836
Non-trainable params: 0
Total mult-adds (M): 5.18
===============================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 24.15
Params size (MB): 1.08
Estimated Total Size (MB): 25.25
===============================================================================================
```

Where can we cut down on parameter count?
