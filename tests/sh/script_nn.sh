#!/usr/bin/expect -f

set hostname [lindex $argv 0];
set first_dim [lindex $argv 1];
set second_dim [lindex $argv 2];
set third_dim [lindex $argv 3];
set poison_rate [lindex $argv 4];
set epochs [lindex $argv 5];

spawn ./short/sshbot.sh $hostname
set timeout 1


expect "~]$"

# Change PathToScript
send "PathToScript/script_nn_serv.sh $first_dim $second_dim $third_dim $poison_rate $epochs\n"

set timeout -1

expect eof

