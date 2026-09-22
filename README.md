<!--
BSD 3-Clause License

Copyright (c) 2026, Honda Research Institute Europe GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SPDX-License-Identifier: BSD-3-Clause-->
# Mirror Eyes

2D eye model with a mirror overlay virtually reflecting an area around a point of interest. 
Requires a display, an image source (e.g. camera), and a point of interest to display the mirror effect and couple it with eye movements. 

Includes a basic face-tracking example, which accesses the local webcam to track and display faces in the camera view. 

## Copyright

```
Copyright 2026

Honda Research Institute Europe GmbH
Carl-Legien-Str. 30
63073 Offenbach/Main
Germany

```
## Setup 

Download the repo
```
git clone git@hri-gitlab.honda-ri.de:robotics/mirror_eyes.git
```

### standard python 

Create a python 3.10 environment and install necessary requirements

#### bash (linux)
```bash
python3.10 -m venv mirror_eyes_env
```

activate it
```bash
source mirror_eyes_env/bin/activate
```

install requirements
```bash
pip install .
```

#### cmd (windows)
```cmd
py -3.10 -m venv mirror_eyes_env
```

activate it
```cmd
mirror_eyes_env\Scripts\activate.bat
```

install requirements
```cmd
pip install .
```

### With uv 

```bash
uv python install 3.10
uv venv 
uv sync
source .venv/bin/activate 
```

## Usage

Run default example application as specified in __main__.py 
```bash
python mirror_eyes
```

or with uv 
```bash 
uv run .
```

In the present configuration this will execute mirror_eyes_app.app, which carries out face-tracking on local camera data to drive the eye model and mirroring. Up to three faces can be tracked simultaneously. 

The behavior of this test-application can be adjusted in the config file `mirror_eyes/config/confic_local_control.py`. For all configuration options see `mirror_eyes/config/config_base.py`. 

In the default app, several behaviours and appearance options can be triggered with the keyboard. Examples: 
- Exit the app press **Esc**. 
- to switch focus between the detected faces or initialization positions, use the keys **1,2, and 3**. To focus on no face, press **0**
- change the pupil size with **+** and **-**
- change the reflection opacity with **y** and **x** 
- cycle through some color examples with **c**
- cycle through some visual effects on the mirror image with **m**
 
### Internal and external control of eye movements

The mirror eyes can currently be used in two modes. 

### Mode 1: Implicit control of eye motion through image source and POI
In the first mode, the eye movements are determined fully by the image source and the current point of interest on that image. Eye movements are then applied such that their movement range is mapped to the resolution of the image source. 

To enable the implicit control mode set the **external_control** parameter of MirrorConfig to **False**.  

### Mode 2: Full control of eye motion
In the second mode, in addition to image source and point of interest, also the location of the eyes on the eye-canvas can be controlled directly. This mode can be preferable when another process already takes care of eye-movement computation. 

To enable the full control mode set the **external_control** parameter of MirrorConfig to **True**.  

## Citation

If you use this software, or results derived from it, in academic or scientific work, please cite the following publication(s):

- Krüger, M., Oshima, Y., & Fang, Y. (2026). Virtual Reflections on a Dynamic 2-D Eye Model Improve Spatial Reference Identification.IEEE Transactions on Human-Machine Systems, vol. 56, no. 2, pp. 203-212, doi: [10.1109/THMS.2026.3651818](https://ieeexplore.ieee.org/document/11362937).
- Krüger, M., Tanneberg, D., Wang, C., Hasler, S., Gienger, M. (2025). Mirror Eyes: Explainable Human-Robot Interaction at a Glance. 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN), Eindhoven, Netherlands, pp. 797-804, doi: [10.1109/RO-MAN63969.2025.11217810](https://ieeexplore.ieee.org/document/11217810).

### BibTex

```bibtex
@ARTICLE{krueger2026virtualreflections,
  author={Matti Kr\"{u}ger and Yutaka Oshima and Yu Fang},
  journal={IEEE Transactions on Human-Machine Systems}, 
  title={Virtual Reflections on a Dynamic 2-D Eye Model Improve Spatial Reference Identification}, 
  year={2026},
  volume={56},
  number={2},
  pages={203-212},
  doi={10.1109/THMS.2026.3651818}}
```

```bibtex
@inproceedings{krueger2025mirroreyes,
  author={Matti Kr\"{u}ger and Daniel Tanneberg and Chao Wang and Stephan Hasler and Michael Gienger},
  booktitle={2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)},
  title={Mirror Eyes: Explainable Human-Robot Interaction at a Glance},
  year={2025},
  pages={797-804},
  doi={10.1109/RO-MAN63969.2025.11217810}
}
```

## Patent Notice

Certain aspects of the technology described or implemented in this repository are subject of patent applications, including the following published applications: 
- US 2025 0291412 A1
- JP 2025 139385 A
- US 2025 0291408 A1
- JP 2025 139333 A

Making this source code available does not grant any rights or licenses under these or any other patents or patent applications related to the work. 
Any rights to use patented technology must be obtained separately from the respective patent holder. 

