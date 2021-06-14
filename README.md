# AdaptiveTimeSteppingSEAS
The main implementation of this thesis is located in the folder "tandem"
To compile and run the simulation:
  - cd tandem
  - mkdir build
  - ccmake ..       # to specify domain dimension, quadrature order ... 
  - cmake -DALIGNMENT=16 -DPETSC_MEMALIGN=16 -DCMAKE_PREFIX_PATH=/path/to/petsc -DLUA_INCLUDE_DIR=/path/to/lua ..
  - make tandem
  - ./app/tandem ../examples/tandem/2d/bp1_sym.toml --petsc -ts_monitor
  
The simulation settings can be changed in tandem/examples/tandem/2d/bp1_sym.toml, tandem/examples/tandem/3d/bp5.toml or any other .toml file

There is also an option to calculate the residual of the Newton iteration and Broyden iteration on the 1st order ODE formulation (works only in 2D)
  - make testJacobian
  - ./app/testJacobian ../examples/tandem/2d/bp1_sym.toml   
