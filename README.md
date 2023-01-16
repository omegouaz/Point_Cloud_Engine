<html>

  <header>     
    <title> Point Cloud Processing Engine | Improve Robotic Vision </title>
  </header>
  
  <body>
       <h1>Abstract</h1>
        <p>Recovering the 3D structure of a scene from images has long been one of the core interests of computer vision. While a considerable amount of work in this area has been devoted to achieve high quality reconstructions regardless of run time, other work has focused on the aspect of real-time reconstruction and its applications. Especially in the last decade the interest in real-time 3D reconstruction systems has increased considerably due to the many applications, industrial as well as scientific, which are enabled by this technology. This thesis fits into this trend by presenting a real-time 3D reconstruction system making use of novel and efficient reconstruction algorithms. Using this system we showcase several innovative applications, focusing in particular on robotic systems (drons, cars) including multi-camera 3D reconstruction systems.</p>

      <h1> The Problem to solve</h1>
      <p>There exists a large number of 3D reconstruction techniques. Unfortunately many of these are not real-time capable and therefore not applicable in real time systems. in this work we will try to make a implimentation of a real-time technique that solves this issue.

      The algorithm presented in this work is attenting to make a reconstruction from unorgnized points using the output of 3D Depth Camera <a href="https://www.intelrealsense.com/depth-camera-d435i/">RealSense D435i</a>
      on general this method is based on Computational geometry it uses the Delaunay triangulation technique to build a set of simplices whise vertices are the original points. Those methods tend to perform well on dense and clean datasets and capable of reconstructing surfaces with boundaries. </p>

  </body>
  
</html>
