for {set i 0} {$i < 56} {incr i} {
	for {set j 0} {$j < 120} {incr j} {
	set name SLICE_X${i}Y${j}
	get_cells  -of_object [get_sites $name]
	}
}