# vaik-detection-trt-inference

Inference with the TensorRT model of the Tensorflow Object Detection API and output the result as a dict in extended
Pascal VOC format.

## Example

![198853671-a868f67f-7105-4ea8-b10b-4362596728c9](https://user-images.githubusercontent.com/116471878/199366179-ebea5174-aea7-4089-91fa-85c5fe85c0e7.png)

## Operation confirmation environment

- tensorrt by docker image
    - g4dn.xlarge
        - nvcr.io/nvidia/tensorrt:22.10-py3
    - jetson xavier nx
        - nvcr.io/nvidia/l4t-tensorflow:r35.1.0-tf2.9-py3

## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-detection-trt-inference.git
```

### Example

```python
import os
import numpy as np
from PIL import Image

from vaik_detection_trt_inference.trt_model import TrtModel

input_saved_model_path = os.path.expanduser('~/output_trt_model/model.trt')
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-detection-dataset/valid/valid_000000000.jpg')).convert('RGB'))

model = TrtModel(input_saved_model_path, classes)
objects_dict_list_list, raw_pred = model.inference([image], score_th=0.2, nms_th=0.5)
```

#### Output

- objects_dict_list

```text

[
    [
      {
        'name': 'eight',
        'pose': 'Unspecified',
        'truncated': 0,
        'difficult': 0,
        'bndbox': {
          'xmin': 564,
          'ymin': 100,
          'xmax': 611,
          'ymax': 185
        },
        'score': 0.9445509314537048
      },
      ・・・
      {
        'name': 'four',
        'pose': 'Unspecified',
        'truncated': 0,
        'difficult': 0,
        'bndbox': {
          'xmin': 40,
          'ymin': 376,
          'xmax': 86,
          'ymax': 438
        },
        'score': 0.38432005047798157
      }
    ],
     ・・・
         'score': 0.38432005047798157
      }
    ],
]
```

- raw_pred

```
[array([[100],
       [100],
        ・・・
       [4, 4, 4, ..., 1, 5, 1],
       [9, 7, 5, ..., 0, 9, 4]], dtype=int32)]

```