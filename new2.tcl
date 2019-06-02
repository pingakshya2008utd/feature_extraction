puts "Name,tile_name,number_of_cells,num_of_pins,num_nets,incoming_nets,outgoing_nets"
#set rt_cong [report_design_analysis $s}
foreach s [get_sites -filter {IS_USED}] {
 #puts $s 
 set tile_name [get_tiles  -of_objects [get_sites $s]]
 set num_cells [llength [get_cells  -of_objects [get_sites $s]]]
 set num_pins [llength [get_pins -of [get_cells -of $s]]]
 set num_incoming_nets [llength [get_pins  -of [get_cells  -of_objects [get_sites $s]] -filter {DIRECTION == IN}]]
 set num_outgoing_nets [llength [get_pins  -of [get_cells  -of_objects [get_sites $s]] -filter {DIRECTION == OUT}]]
 set nets [get_nets -of_objects [get_sites $s]]
 lsort -unique $nets
 set num_nets  [llength $nets]
puts [format "%s,%s,%d,%d,%d,%d,%d" $s $tile_name $num_cells $num_pins $num_nets $num_incoming_nets $num_outgoing_nets]
 }