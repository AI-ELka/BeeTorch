import os

epochs = int(input("How many epochs : "))

subdomain_name = ["habenaria","porsche","rouget","piranha","mulet","requin","atlas","ardennes","raie","perdrix","quiscale","rouloul","sitelle"]
bad_sub = ["barbeau","epipactis","sole"]

subdomain_name = subdomain_name+bad_sub
hostname_known = [subdomain_name[i]+".polytechnique.fr" for i in range(len(subdomain_name))]
hostname_known = hostname_known[:21]

training_set = [
    [19,0,0],
    #[14,0,0],
    #[10,0,0],
    [23,0,0],
    [24,0,0],
    [17,1,.1],
    [18,1,.1],
    #[14,1,.1],
    [15,1,.1],
    [19,1,.1],
    [20,1,.1],
    [21,1,.1],
    [13,1,.1],
    [7,1,.1],
    #[9,1,.1],
    [22,1,.1],
    [23,1,.1],
]
training_set.sort()
training_set.reverse()
for i in range(min(len(hostname_known),len(training_set))):
    #os.system("gnome-terminal -- '/bin/bash -c \"./script.sh "+hostname_known[0]+" 1 0 0 200\" '")
    host = hostname_known[i]
    a,b,c = training_set[i]
    title = ""
    title += str(a)
    if b==1:
        title+=" Flip "+str(int(a*100))
    os.system(f"gnome-terminal --title=\"{title}\" -- bash -c \"cd ~;./script.sh {host} {a} {b} {c} {epochs}; exec bash -i \"")

if len(hostname_known)-len(training_set)<0:
    print(f"Not enough servers available for training, you need {len(training_set)-len(hostname_known)} more servers!")

input("wait")