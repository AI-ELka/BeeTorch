#!/usr/bin/expect -f

set hostname [lindex $argv 0];

spawn ./short/sshbot.sh $hostname
set timeout 2


expect "login"
send "killall python -9\n"

set timeout 1

send "killall bash -9\n"

set timeout 1

expect eof