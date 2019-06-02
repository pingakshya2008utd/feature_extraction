set device       "xcku085"
set package      "-flvf1924"
set speed        "-1LV-i"
set part         $device$package$speed

for {set i 1} {$i < 5} {incr i} {
    puts $i
	
	set synth_var {}
	set place_var {}
	set route_var {}
	set name1 "C:/ddr_read/ISPD2016/FPGA-example"
	set dcpname "/design.dcp"
	set placename "/post_place.dcp"
	set routename "/post_route.dcp"
	append synth_var $name1$i$dcpname
	append place_var $name1$i$placename
	append route_var $name1$i$routename
	puts $synth_var
	puts $place_var
	puts $route_var
	set_part  xcku085-flvf1924-1LV-i
	open_checkpoint $synth_var
	opt_design
	#Place design using the bookshelf
	place_design
	write_checkpoint -force $place_var

	#Route design
	route_design
	write_checkpoint -force $route_var
	#Routing Report
	report_route_status
	#close_project
}

#for {set i 1} {$i < 3} {incr i}{

	#puts $i
	
	# set var2 {}
	# set name1 "C:/ddr_read/ISPD2016/FPGA-example"
	# set dcpname "/design.dcp"
	# append var2 $name1$i$dcpname
	# puts $var2 

	# open_checkpoint $var2
	# synth_design -mode out_of_context -part $part
	# opt_design
	# #Place design using the bookshelf
	# place_design
	# write_checkpoint -force post_place.dcp

	# #Route design
	# route_design
	# write_checkpoint -force post_route.dcp
	# #Routing Report
	# report_route_status
	# close_design

#}
