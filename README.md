## Guidance to use
If you are using colab, you should firstly clone this repository:
```python
!rm -rf /content/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition
!git clone https://github.com/Hewitt6/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition.git
```
Then download the ucf101 dataset from its website
```python
!wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
!unrar x -Y UCF101.rar
```
Next, you need to extract frames in jpg from the avi videos and creat a n_frames files for each directories:
```python
!mkdir jpg
!python /content/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition/utils/video_jpg_ucf101_hmdb51.py /content/UCF-101 /content/jpg
!python /content/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition/utils/n_frames_ucf101_hmdb51.py /content/jpg
```
Now you can start training the model. There are three recommended options for models shufflenet, causalshuf(our best model) and twowayshuf. slowfastshuf, slowfastcausalshuf are also available implementations, but not recommended.
```
!python /content/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition/main.py --root_path ~/ \
	--video_path /content/jpg \
	--annotation_path /content/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition/annotation_UCF101/ucf101_01.json \
	--result_path /content/3D-Causal-ShuffleNet-in-Dual-Paths-for-Efficient-Action-Recognition/results \
	--dataset ucf101 \
	--n_classes 101 \
	--model causalshuf \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 32 \
	--downsample 1 \
	--batch_size 16 \
	--n_threads 16 \
	--n_val_samples 1 
```
## Architecture
```python
TwoPathwayShuffleNet(
  (slow_conv): Sequential(
    (0): Conv3d(3, 24, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
    (1): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (slow_maxpool): MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)
  (fast_avgpool): AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
  (fast_conv): Sequential(
    (0): Conv3d(3, 24, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1), bias=False)
    (1): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (slow_shufflenet): ShuffleNet(
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv3d(24, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (shortcut): AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
      )
      (1): Bottleneck(
        (conv1): Conv3d(120, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv3d(120, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv3d(120, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv3d(120, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (shortcut): AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
      )
      (1): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (5): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (6): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (7): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv3d(240, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (shortcut): AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
      )
      (1): Bottleneck(
        (conv1): Conv3d(480, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 480, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv3d(480, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 480, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv3d(480, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 480, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (fast_shufflenet): ShuffleNet(
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv3d(24, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (shortcut): AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
      )
      (1): Bottleneck(
        (conv1): Conv3d(120, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv3d(120, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv3d(120, 30, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(30, 30, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=30, bias=False)
        (bn2): BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(30, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv3d(120, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (shortcut): AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
      )
      (1): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (5): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (6): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (7): Bottleneck(
        (conv1): Conv3d(240, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=60, bias=False)
        (bn2): BatchNorm3d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(60, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv3d(240, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (shortcut): AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
      )
      (1): Bottleneck(
        (conv1): Conv3d(480, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 480, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv3d(480, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 480, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv3d(480, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn1): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(120, 120, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=120, bias=False)
        (bn2): BatchNorm3d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv3d(120, 480, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3, bias=False)
        (bn3): BatchNorm3d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=960, out_features=101, bias=True)
  )
)
Total number of trainable parameters:  613661
Total number of FLOPs:  110313640.0
```
