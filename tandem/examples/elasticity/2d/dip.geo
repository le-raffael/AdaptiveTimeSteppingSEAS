DefineConstant[ h = {0.1, Min 0, Max 10, Name "Resolution" } ];
DefineConstant[ hf = {0.05, Min 0, Max 10, Name "Resolution at fault-surface intersection" } ];
DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];
DefineConstant[ L = {2.0, Min 0, Max 10, Name "L" } ];
DefineConstant[ H = {1.0, Min 0, Max 10, Name "H" } ];

dip_rad = -dip * Pi / 180.0;

Point(1) = {L, 0, 0, h};
Point(2) = {0, 0, 0, hf};
Point(3) = {-H * Cos(dip_rad)/Sin(dip_rad), -H, 0, hf};
Point(4) = {L, -H, 0, h};
Point(5) = {-L, -H, 0, h};
Point(6) = {-L, 0, 0, h};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {3, 5};
Line(6) = {5, 6};
Line(7) = {6, 2};
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {2, 5, 6, 7};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Curve(1) = {1, 7};
Physical Curve(3) = {2};
Physical Curve(5) = {3, 4, 5, 6};
Physical Surface(1) = {1, 2};
Mesh.MshFileVersion = 2.2;