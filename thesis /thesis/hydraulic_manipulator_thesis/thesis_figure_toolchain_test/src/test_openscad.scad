// OpenSCAD verification: hydraulic cylinder components
$fn = 60;

module cylinder_body(d = 30, h = 60) {
    difference() {
        cylinder(d = d, h = h, center = true);
        // bore hole
        cylinder(d = 15, h = h + 1, center = true);
    }
}

module piston_rod(d = 14, l = 80) {
    cylinder(d = d, h = l, center = false);
}

module clevis(w = 20, h = 30, pin_d = 6) {
    difference() {
        translate([-w/2, -5, 0]) cube([w, 10, h]);
        translate([0, 0, h/2]) rotate([90, 0, 0]) 
            cylinder(d = pin_d, h = 12, center = true);
    }
}

translate([0, 0, -30]) cylinder_body();
translate([0, 0, 0]) piston_rod();
translate([0, 0, 80]) rotate([0, 0, 0]) clevis();
