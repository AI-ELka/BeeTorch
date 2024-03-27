#!/usr/bin/expect -f

set first_dim [lindex $argv 0];
set second_dim [lindex $argv 1];
set third_dim [lindex $argv 2];
set poison_rate [lindex $argv 3];
set epochs [lindex $argv 4];

spawn ./launchnn.sh

expect >>

send "import tests.model\n"

expect "poison rate"

send "$poison_rate\n"
expect "epochs"
send "$epochs\n"

expect "hidden dimension "
send "$first_dim\n$second_dim\n$third_dim\n"

set timeout -1

expect eof