Workspace Format ID  = 3
[Header Data]:
   Format ID = 1
   PrismSPECT Version_7.1.0
[End Header Data]

[Material Elements Parameters]:
   Format ID = 6
#
 Default ATBASE directory      = ATBASE_6.2
 Default PROPACEOS directory   = PROPACEOS_DATA_5.1.0
#
  Number of Atomic Elements = 2
#
  Parameters for Atomic Element =  Xenon
#
   Atomic number            = 54
   Isotope index            = -1
   Atomic weight            = 131.3
   Number fraction          = 0.013
   Element Symbol           = Xe
   ATM specifier ID         = 0
   Default atomic model ID  = 1
   ATM filepath             =   
   Tabular opacity filepath =   
   Oplib opacity filepath   = 
#
#
  Parameters for Atomic Element =  Hydrogen
#
   Atomic number            = 1
   Isotope index            = -1
   Atomic weight            = 1.00797
   Number fraction          = 0.987
   Element Symbol           = H
   ATM specifier ID         = 0
   Default atomic model ID  = 1
   ATM filepath             =   
   Tabular opacity filepath =   
   Oplib opacity filepath   = 
#
#
[End Material Elements Parameters]

[Simulation Type Parameters]:
   Format ID = 5
#
   Simulation class ID           = 1
   Geometry model ID             = 1
   Elec distrib specification ID = 0
   Hot electron specification ID = 0
   Density specification ID      = 0
   Size specification ID         = 0
   Ext rad src model ID          = 1
   Ext rad src sides ID          = 0
   Ion beam model ID             = 0
   Pressure model type           = 0
   Pressure model fixed temp.    = 273.15
   Pressure EOS filename         = 
#
[End Simulation Type Parameters]

[Steady-State Plasma Parameters]:
   Format ID = 8
#
   Indep. var. #1 name          = Plasma Temperature
   Indep. var. #2 name          = Ion Density
   Indep. var. #1 ID            = 1
   Indep. var. #2 ID            = 2
   Element being varied         = 
   Plasma temperature           = 10
   Plasma temperature units ID  = 0
   Plasma density               = 0.01
   Plasma density units ID      = 0
   Plasma size                  = 0.4
   Plasma size units ID         = 0
   Rad drive temperature        = 1
   Rad drive temp units ID      = 0
   Rad spectral temperature     = 1
   Rad spectral temp units ID   = 0
   Line drive min photon energy = 0
   Line drive max photon energy = 0
   VisRad file path             =   
   Use size-density model       = 0
   Size-density model ID        = 1
   Hot electron fraction        = 0
   Hot electron density         = 0
   Hot electron temperature     = 1000
   Hot electron temp units ID   = 0
   Limit range elec distrib     = 0
   Min hot elec distrib range   = 0.0001
   Max hot elec distrib range   = 500000
   Hot elec distrib expression  = 
   Hot elec histogram file      = 
   Ion beam energy              = 1
   Ion beam current             = 1
 [table format=1]:    Steady-State Plasma Temperatures Grid:
  # table rows = 10
  # table cols = 1
Plasma Temperature
 2.00000e+01   4.00000e+01   6.00000e+01   8.00000e+01   1.00000e+02   1.20000e+02   1.40000e+02   1.60000e+02   1.80000e+02   2.00000e+02  
 [table format=1]:    Steady-State Plasma Density Grid:
  # table rows = 10
  # table cols = 1
Density
 1.00000e+18   5.00000e+18   1.00000e+19   2.50000e+19   5.00000e+19   7.50000e+19   1.00000e+20   2.50000e+20   5.00000e+20   1.00000e+21  
 [table format=1]:    Steady-State Plasma Size Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Element Number Fraction Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Radiation Drive Temperature Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Radiation Spectral Temperature Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Hot Electron Fraction Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Hot Electron Density Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Hot Electron Temperature Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Radiation Drive Flux
  # table rows = 0
  # table cols = 0
#
[End Steady-State Plasma Parameters]

[Time-Dependent Plasma Parameters]:
   Format ID = 9
#
   Plasma temperature           = 1.00000e+01
   Plasma temperature units ID  = 0
   Plasma density               = -1.00000e+00
   Plasma density units ID      = 0
   Plasma size                  = 1.00000e-03
   Plasma size units ID         = 0
   Rad drive temperature        = 1.00000e+00
   Rad drive temp units ID      = 0
   Rad spectral temperature     = 1.00000e+00
   Rad spectral temp units ID   = 0
   Line drive min photon energy = 0.00000e+00
   Line drive max photon energy = 0.00000e+00
   VisRad file path             =   
   Use plasma temp table        = 0
   Use plasma density table     = 0
   Use plasma size table        = 0
   Use rad drive temp table     = 0
   Use rad spectral temp table  = 0
   Use size-density model       = 0
   Size-density model ID        = 1
   Init populs model ID         = 1
   Init populs temperature      = 2.50000e-02
   Init populs filepath         = 
   Hot electron fraction        = 0.00000e+00
   Hot electron density         = 0.00000e+00
   Hot electron temperature     = 1.00000e+03
   Hot electron temp units ID   = 0
   Use hot elec fraction table  = 0
   Use hot elec density table   = 0
   Use hot electron temp table  = 0
   Limit range elec distrib     = 0
   Min hot elec distrib range   = 1.00000e-04
   Max hot elec distrib range   = 5.00000e+05
   Hot elec distrib expression  = 
   Hot elec histogram file      = 
   Ion beam energy              = 1.00000e+00
   Ion beam current             = 1.00000e+00
 [table format=1]:    Simulation Time Grid:
  # table rows = 1
  # table cols = 1
Simulation Times:
 0.00000e+00  
 [table format=1]:    Time-Dependent Plasma Temperatures Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Time-Dependent Plasma Density Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Time-Dependent Plasma Size Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Time-Dependent Radiation Drive Temperature Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Time-Dependent Radiation Spectral Temperature Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Hot Electron Fraction Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Hot Electron Density Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Steady-State Hot Electron Temperature Grid:
  # table rows = 0
  # table cols = 0
 [table format=1]:    Time-Dependent Radiation Drive Flux
  # table rows = 0
  # table cols = 0
#
[End Time-Dependent Plasma Parameters]

[Atomic Processes Parameters]:
   Format ID = 17
#
   Population model ID       = 1
   Coll Exc/Deexc mult.      = 1.00000e+00
   Coll Iz/Rec mult.         = 1.00000e+00
   Photo Exc/Stim emis mult. = 1.00000e+00
   Photo Iz/Stim rec mult.   = 1.00000e+00
   Spontaneous emis mult.    = 1.00000e+00
   Radiative rec mult.       = 1.00000e+00
   Autoiz/Diel rec mult.     = 1.00000e+00
   Min. Osc. Strength for grid  = 1.00000e-03
   Num pts in continuum      = 1000
   Num pts per BB transition = 13
   Doubly excited trans.     = 0
   Cont lowering model ID    = 1
   Energy scale factor       = 1.00000e+00
   Dense plasma              = 0
   Turn off cont. lowering   = 0
   Apply accel. convergence  = 0
   Include inner transitions = 0
   Max. # of bound electrons = 100
   Min. # of bound electrons = 11
   Max. principal q. num.    = 4
   Include degeneracy corr.  = 0
   Include Coulomb corr.     = 0
   Use the Prism solver      = 0
   Num. timesteps for solver = 1000
   Compute steady-state pops = 0
   Line width mods formatID  = 1
   Line profile data type    = 0
   Line profile data file    =  
   Line width units          = 0
   Num line width settings   = 0
   B-B modifiers format ID  = 1
   Number of modified lines = 0
   Wavelength modifier mode = 0
   Wavelength units ID      = 0
   Osc. str. modifier mode  = 0
#
[End Atomic Processes Parameters]

[Spectral Parameters]:
   Format ID = 3
#
   Min photon energy            = 7.00000e+02
   Max photon energy            = 1.10000e+03
   Num continuum pts            = 200
   Min line strength            = 1.00000e-03
   Num pts per line transition  = 13
   Backlighter model ID         = 0
   Backlighter temperature      = 1.00000e+03
   Backlighter line temperature = 1.00000e+02
   Backlighter line energy min  = 1.00000e+00
   Backlighter line energy max  = 1.00000e+04
   Backlighter file path        =   
   Backlighter photon units     = 0
   Backlighter intensity units  = 0
   Backlighter intensity mult.  = 1.00000e+00
#
[End Spectral Parameters]

[Output Control Parameters]:
   Format ID = 4
#
   Write pops/spectra to files     = 1
  Number of opacity components     = 0
#
[End Output Control Parameters]

[Dialog Settings Parameters]:
   Format ID = 1
#
[End Dialog Settings Parameters]

[Line Intensity Parameters]:
   Format ID = 3
#
   Plot quantity type           = 0
   SpectralBand formatID        = 4
   Number of Line Bands <Indiv Trans> = 0
   Number of Line Bands <Intg and Fitted> = 0
   Number of Line Ratios <Indiv Trans> = 0
   Number of Line Ratios <Intg. and Fitted> = 0
#
[End Line Intensity Parameters]


[End of Workspace File]

