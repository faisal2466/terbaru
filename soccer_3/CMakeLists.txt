cmake_minimum_required(VERSION 2.8.3)
project(soccer_3)

find_package(catkin REQUIRED)

catkin_package()

# Optional: untuk validasi launch files
find_package(roslaunch REQUIRED)

# Install folders: urdf, launch, meshes, config
foreach(dir config launch meshes urdf)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir}
    FILES_MATCHING PATTERN "*.*"
  )
endforeach(dir)

