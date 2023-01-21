<p align="center">
  <a href="">
    <img src="logo/logo.png" alt="Logo" width=72 height=72>
  </a>

  <h3 align="center">Point Cloud Reconstruction Engine </h3>
</p>


## Table of contents

- [Abstract](#abstract)
- [Problem](#problem)
- [Full Architecture](#full_architecture)
- [Bugs and feature requests](#bugs-and-feature-requests)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)


## abstract

Recovering the 3D structure of a scene from images has long been one of the core interests of computer vision. While a considerable amount of work in this area has been devoted to achieve high quality reconstructions regardless of run time, other work has focused on the aspect of real-time reconstruction and its applications. Especially in the last decade the interest in real-time 3D reconstruction systems has increased considerably due to the many applications, industrial as well as scientific, which are enabled by this technology. This thesis fits into this trend by presenting a real-time 3D reconstruction system making use of novel and efficient reconstruction algorithms. Using this system we showcase several innovative applications, focusing in particular on robotic systems (drons, cars) including multi-camera 3D reconstruction systems.

## problem

There exists a large number of 3D reconstruction techniques. Unfortunately many of these are not real-time capable and therefore not applicable in real time systems. in this work we will try to make a implimentation of a real-time technique that solves this issue.
The algorithm presented in this work is attenting to make a reconstruction from unorgnized points using the output of 3D Depth Camera <a href="https://www.intelrealsense.com/depth-camera-d435i/">RealSense D435i</a> on general this method is based on Computational geometry it uses the Delaunay triangulation technique to build a set of simplices whise vertices are the original points. Those methods tend to perform well on dense and clean datasets and capable of reconstructing surfaces with boundaries.
>>>>>>> 4a38203 (adding logo)

## full_architecure

Some text

```text
source_code_project/
└── pcp_engine/
    ├── engine/
    │   ├── file1
    │   └── file2
    └── docs/
    │   ├── file3
    │   └── file4
    │____main.py

```


## Bugs and feature requests

Have a bug or a feature request? Please first read the [issue guidelines](https://reponame/blob/master/CONTRIBUTING.md) and search for existing and closed issues. If your problem or idea is not addressed yet, [please open a new issue](https://reponame/issues/new).

## Contributing

Please read through our [contributing guidelines](https://reponame/blob/master/CONTRIBUTING.md). Included are directions for opening issues, coding standards, and notes on development.

Moreover, all HTML and CSS should conform to the [Code Guide](https://github.com/mdo/code-guide), maintained by [Main author](https://github.com/usernamemainauthor).

Editor preferences are available in the [editor config](https://reponame/blob/master/.editorconfig) for easy use in common text editors. Read more and download plugins at <https://editorconfig.org/>.

<<<<<<< HEAD
## Creators

- <https://github.com/OussamaMegouas>

<<<<<<< HEAD
## Copyright and license

Code and documentation copyright 2022-2023 the authors. Code released under the [MIT License](https://github.com/OussamaMegouas/pcp_engine/blob/main/LICENSE).


=======
## Copyright and license

Open3D: A Modern Library for 3D Data Processing [MIT License](https://github.com/isl-org/Open3D).

Intel® RealSense™ SDK 2.0 is a cross-platform library for Intel® RealSense™ depth cameras (D400 & L500 series and the SR300) This project only uses the python wrapper [Apache License](https://github.com/IntelRealSense/librealsense).
