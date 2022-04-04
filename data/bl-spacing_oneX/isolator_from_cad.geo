Merge "isolator.brep";

If(Exists(size))
    basesize=size;
Else
    basesize=1;
EndIf

If(Exists(blratio))
    boundratio=blratio;
Else
    boundratio=1.0;
EndIf

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize*2;   // background mesh size upstream of the nozzle
isosize = basesize;       // background mesh size in the isolator
nozzlesize = basesize/2;       // background mesh size in the isolator
cavitysize = basesize/2.; // background mesh size in the cavity region

Physical Volume('fluid') = {1};
Physical Volume('wall surround') = {2};
Physical Volume('wall insert') = {3};

Physical Surface('inflow') = {1};
Physical Surface('outflow') = {6};
Physical Surface('isothermal') = {2:5,7,10,13:20};
Physical Surface('fluid wall interface') = {8,9,11,12};
Physical Surface('wall far-field') = {21:24};

// Create distance field from curves, excludes cavity
Field[1] = Distance;
Field[1].SurfacesList = {2:5,7:9,15:20};
Field[1].NumPointsPerCurve = 1000;

//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = nozzlesize / boundratio;
Field[2].SizeMax = isosize;
Field[2].DistMin = 0.2;
Field[2].DistMax = 5;
Field[2].StopAtDistMax = 1;

// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].SurfacesList = {10:14};
Field[11].NumPointsPerCurve = 1000;

//Create threshold field that varrries element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratio;
Field[12].SizeMax = cavitysize;
Field[12].DistMin = 0.2;
Field[12].DistMax = 5;
Field[12].StopAtDistMax = 1;

// Create distance field from curves, inside wall only
Field[13] = Distance;
Field[13].SurfacesList = {25:30};
Field[13].NumPointsPerCurve = 1000;

//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = cavitysize / boundratio;
Field[14].SizeMax = cavitysize;
Field[14].DistMin = 0.2;
Field[14].DistMax = 5;
Field[14].StopAtDistMax = 1;

// nozzle_start = 270;
// nozzle_end = 300;
// //  background mesh size in the isolator (downstream of the nozzle)
// Field[3] = Box;
// Field[3].XMin = nozzle_end;
// Field[3].XMax = 1000.0;
// Field[3].YMin = -1000.0;
// Field[3].YMax = 1000.0;
// Field[3].VIn = isosize;
// Field[3].VOut = bigsize;
// 
// // background mesh size upstream of the inlet
// Field[4] = Box;
// Field[4].XMin = 0.;
// Field[4].XMax = nozzle_start;
// Field[4].YMin = -1000.0;
// Field[4].YMax = 1000.0;
// Field[4].VIn = inletsize;
// Field[4].VOut = bigsize;
// 
// // background mesh size in the nozzle throat
// Field[5] = Box;
// Field[5].XMin = nozzle_start;
// Field[5].XMax = nozzle_end;
// Field[5].YMin = -1000.0;
// Field[5].YMax = 1000.0;
// Field[5].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
// Field[5].VIn = nozzlesize;
// Field[5].VOut = bigsize;
// 
// // background mesh size in the cavity region
// cavity_start = 650;
// cavity_end = 730;
// Field[6] = Box;
// Field[6].XMin = cavity_start;
// Field[6].XMax = cavity_end;
// Field[6].YMin = -1000;
// Field[6].YMax = -3;
// Field[6].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
// Field[6].VIn = cavitysize;
// Field[6].VOut = bigsize;

// take the minimum of all defined meshing fields
Field[100] = Min;
// Field[100].FieldsList = {2, 3, 4, 5, 6, 12, 14};
Field[100].FieldsList = {2,12,14};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

// Mesh.CharacteristicLengthMin = basesize;
Mesh.CharacteristicLengthMax = basesize;

// Millimeters to meters
Mesh.ScalingFactor = 0.001;

//Mesh.Smoothing = 3;
