#!/usr/bin/expect -f

set hostname [lindex $argv 0];
set degree [lindex $argv 1];
set poison [lindex $argv 2];
set poisonRate [lindex $argv 3];
set epochs [lindex $argv 4];

spawn ./launchssh.sh $hostname
set timeout 1


expect "~]$"
send "./script.sh $degree $poison $poisonRate $epochs\n"

set timeout -1

expect eof

