num_outs=$(find . -maxdepth 1 -type f -name 'out_*.txt' | wc -l)

if (( $num_outs > 0 )); then
	while true; do
	    read -p "Do you want to remove previous outputs? [Y/n]" yn
	    case $yn in
        	[Yy]* ) rm out_*.txt; rm *-infinite.p; break;;
	        [Nn]* ) break;;
	        * ) echo "Please answer yes or no.";;
	    esac
	done
fi

for start in `seq 0.0075 0.01 0.1`; 
do 
	final=$(echo "$start + 0.005" | bc)
	
	start_formated=$(printf "%3.5f\n" $start)
	end_formated=$(printf "%3.5f\n" $final)
	
	sed -i 's/steps = .*/steps = 1/g' network.py
	sed -i "s/gateway_distance_min = .*/gateway_distance_min = $start_formated/g" nano_parameters.py
	sed -i "s/gateway_distance_max = .*/gateway_distance_max = $end_formated/g" nano_parameters.py
	
	echo "Min distance between gateways = $start_formated"
	echo "Max distance between gateways = $end_formated"

	for policy in MDP; do # AS OSHP RP HHP CSP NSP TP; do
		sed "s/policy_name = .*/policy_name = '$policy'/g" network.py | grep "policy_name = "
		/usr/bin/python3 network.py
	done
	
done
