# scan_correction

This script is associated with a specific msem experiment.
The msem experiment comprises 16 "shift".
For each "shift", a set of 7 mFOVs is acquired:

  O  O
  
O  O  O

  O  O
  
A mFOV is made of 61 sFOVs.
There is a translation offset in the x axis between shift number 1 and shift number 2.
There is the same translation offset between shift n and shift (n+1)
The translation offset is 1/16 of the width of an sFOV, which is about 1 micrometer.
The translation offset between shift 0 and shift 8 is 8/16 = 1/2 of the width of a sFOV
As a first approximation, it could be assumed that the right half of a sFOV
does not have any distortion. This second half could be taken as a ground truth
to compute the scan correction

To find the scan distortion, a vertical 10 pixel wide band from the translated image
is fit to the non-translated image.

![Naming convention](scan_correction_naming.jpg?raw=true "Naming convention")